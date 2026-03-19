The MuLaConf Package
====================================

Package Wrapper
---------------

.. autoclass:: mulaconf.icp_wrapper.ICPWrapper
   :members: fit,calibrate,predict

Inductive Conformal Predictor
-----------------------------

.. autoclass:: mulaconf.icp_predictor.InductiveConformalPredictor
   :members: calibrate,predict

Prediction Regions
------------------

.. autoclass:: mulaconf.prediction_regions.PredictionRegions
   :special-members: __call__
   :members: evaluate

