from typing import Dict, Type, Any
import logging
from urm.config import Config
from .model import Model

_MODEL_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(name: str):
    def decorator(cls: Type[Model]):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered!")
        _MODEL_REGISTRY[name] = cls
        logging.info(f"model {name} is registered")
        return cls

    return decorator


def get_model_class(name: str) -> Type[Model]:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def create_model_from_config(config: Config) -> Model:
    model_type = config.reward.prediction_model
    if not model_type:
        raise ValueError("Config must contain 'type' field")
    model_cls = get_model_class(model_type)
    model = model_cls(config.reward.prediction_model_configs)
    return model
