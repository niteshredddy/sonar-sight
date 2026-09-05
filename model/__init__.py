"""Model package — sonar debris detection pipeline."""

from .inference import SonarInferencePipeline
from .preprocessing import preprocess_sonar, CFARTransform
from .confidence_rescorer import GeometricRescorer, get_rescorer

__all__ = [
    "SonarInferencePipeline",
    "preprocess_sonar",
    "CFARTransform",
    "GeometricRescorer",
    "get_rescorer",
]
