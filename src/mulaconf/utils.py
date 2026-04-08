import numpy as np
import pandas as pd
import torch
import warnings
import hashlib
import json


from typing import Union


def _check_multihot_labels(
        labels: Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
) -> Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]:
    """
    Checks if the labels are in a binary multi-hot format (containing only 0s and 1s).

    This function validates the values WITHOUT forcing a conversion to PyTorch tensors.
    It returns the input object unmodified.

    Parameters
    ----------
    labels : Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The input labels to check.

    Returns
    -------
    Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series]
        The original labels object, validated.

    Raises
    ------
    ValueError
        If labels contain values other than 0 or 1.
    """

    if isinstance(labels, (pd.DataFrame, pd.Series)):
        vals = labels.values
    elif isinstance(labels, torch.Tensor):
        vals = labels
    else:
        vals = np.asarray(labels)

    if isinstance(vals, torch.Tensor):
        unique_vals = torch.unique(vals)
        unique_vals = unique_vals.detach().cpu().numpy()
    else:
        unique_vals = np.unique(vals)

    for val in unique_vals:
        if val != 0 and val != 1:
            raise ValueError(f"Labels must be binary (0 and 1). Found value: {val}")

    return labels


def _is_tensor(data: Union[torch.Tensor, np.ndarray, list, pd.DataFrame, pd.Series],
               dtype=torch.float32) -> torch.Tensor:
    """
    Converts input data to a PyTorch tensor.
    Handles PyTorch Tensors, NumPy arrays, Pandas DataFrame/Series,
    standard lists, and Scikit-Learn friendly array-likes.

    Parameters
    ----------
    data : array-like
        The input data to convert.
    dtype : torch.dtype, optional, default=torch.float32
        The desired data type of the output tensor.

    Returns
    -------
    torch.Tensor
        The converted PyTorch tensor.
    """

    if torch.is_tensor(data):
        return data.to(dtype=dtype)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    elif hasattr(data, "values"):
        data = data.values

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    return torch.tensor(data, dtype=dtype)


def _normalize_device(dev: Union[torch.device, str, None]) -> torch.device:
    if dev is None:
        return torch.device("cpu")
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")

    return dev


def _fingerprint_model(strategy, kwargs):
    """
    Create a deterministic fingerprint based on model class and hyperparameters only.
    """

    class SklearnEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if callable(obj):
                return getattr(obj, '__name__', str(type(obj).__name__))
            return super().default(obj)

    def get_stable_params(estimator):
        """Extract only configuration params, not fitted state."""
        if not hasattr(estimator, 'get_params'):
            return {'__class__': type(estimator).__name__}

        params = {}
        for key, value in estimator.get_params(deep=False).items():
            if hasattr(value, 'get_params'):
                params[key] = get_stable_params(value)
            else:
                params[key] = value

        return {'__class__': type(estimator).__name__, '__params__': params}

    fingerprint_data = {
        'model': get_stable_params(strategy),
        'kwargs': kwargs
    }

    try:
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, cls=SklearnEncoder)
    except Exception:
        fingerprint_str = str(fingerprint_data)

    return hashlib.sha256(fingerprint_str.encode()).hexdigest()