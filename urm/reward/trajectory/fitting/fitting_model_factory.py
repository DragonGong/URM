from typing import Dict, Type, Any

from urm.config import Config
from .fitting import Fitting  # 注意：确保你已定义或导入 Fitting 基类

_FITTING_REGISTRY: Dict[str, Type[Fitting]] = {}


def register_fitting(name: str):
    def decorator(cls: Type[Fitting]):
        if name in _FITTING_REGISTRY:
            raise ValueError(f"Fitting '{name}' is already registered!")
        _FITTING_REGISTRY[name] = cls
        return cls

    return decorator


def get_fitting_class(name: str) -> Type[Fitting]:
    if name not in _FITTING_REGISTRY:
        raise ValueError(f"Unknown fitting: {name}. Available: {list(_FITTING_REGISTRY.keys())}")
    return _FITTING_REGISTRY[name]


def create_fitting_from_config(config: Config) -> Fitting:
    fitting_type = config.reward.prediction_model
    if not fitting_type:
        raise ValueError("Config must contain 'prediction_model' field")
    fitting_cls = get_fitting_class(fitting_type)
    fitting = fitting_cls(**config.reward.fitting_model_configs)
    return fitting
