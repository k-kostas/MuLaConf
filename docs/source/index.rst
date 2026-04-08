MuLaConf Package
================================

A flexible Python package for **Conformal Prediction (CP)** in **Multi-label** classification settings.
It implements the **Powerset Scoring** approach :ref:`[3] <ref-3>` using both the **Mahalanobis**
distance :ref:`[1] <ref-1>` and the standard **Euclidean Norm** :ref:`[4] <ref-4>` as nonconformity measures, and applies
**Structural Penalties** to provide more informative prediction sets, based on Hamming distance
and label-set cardinality :ref:`[2] <ref-2>`. Designed for efficiency, it handles model training, calibration,
and the on-the-fly update of structural penalty weights or distance measures without the need
for model retraining. This package bridges **Scikit-Learn** (for the underlying classifiers)
and **PyTorch** (for efficient tensor computations and GPU acceleration).


Key Features
------------
* **Multi-label Conformal Prediction**: Provides sets of label-sets with guaranteed coverage under the assumption of data exchangeability.
* **Powerset Scoring**: Explicitly assigns p-values to all possible label-sets.
* **Distance Measures**: Supports both the **Mahalanobis** distance and the standard **Euclidean Norm** in the error vector space.
* **Structural Penalties**: Incorporates Hamming and Cardinality penalties to produce more informative prediction sets.
* **Post-training Penalty Updates**: Modify penalty weights after fitting, with no need to retrain the model or recalculate the covariance matrix.
* **Automatic Classifier Switching**: Replace the underlying classifier (e.g., from :class:`~sklearn.ensemble.RandomForestClassifier` to :class:`~sklearn.neighbors.KNeighborsClassifier`) and let the wrapper handles retraining automatically.
* **Compatible with any model**: Provides a wrapper (ICPWrapper) for any sklearn multi-label classifier (e.g., :class:`~sklearn.multioutput.MultiOutputClassifier`, :class:`~sklearn.multioutput.ClassifierChain`) plus a model agnostic InductiveConformalPredictor.
* **GPU Support**: Offloads heavy matrix computations to CUDA devices.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started

   documentation

   citing

   references