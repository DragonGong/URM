from .fitting import Fitting
from .fitting_model_factory import register_fitting, create_fitting_from_config, get_fitting_class
from .polynomial import Polynomial

__all__ = [
    "get_fitting_class",
    "create_fitting_from_config",
    "register_fitting",
    "Polynomial",
    "Fitting",
]
