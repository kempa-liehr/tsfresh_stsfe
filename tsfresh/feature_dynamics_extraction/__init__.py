"""
The :mod:`tsfresh.feature_extraction` module contains methods to extract the features from the time series
"""

from tsfresh.feature_dynamics_extraction.feature_dynamics_extraction import (
    extract_feature_dynamics,
)
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)
