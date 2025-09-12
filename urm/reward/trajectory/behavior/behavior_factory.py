# ===== 全局工厂 =====
from urm.config import Config


class BehaviorFactory:
    _registry = {}

    def __init__(self, config: Config.RewardConfig.BehaviorConfigs):
        self.config = config

    @classmethod
    def register(cls, name: str):
        """注册装饰器"""

        def decorator(klass):
            cls._registry[name] = klass
            return klass

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """根据名字创建对象"""
        if name not in cls._registry:
            raise ValueError(f"Behavior '{name}' not found. 已注册的: {list(cls._registry.keys())}")
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def available(cls):
        """查看有哪些可用的行为类"""
        return list(cls._registry.keys())

    @classmethod
    def get_all_behaviors(cls):
        """
        获取所有注册的行为类
        返回一个 dict: {name: class}
        """
        return dict(cls._registry)
