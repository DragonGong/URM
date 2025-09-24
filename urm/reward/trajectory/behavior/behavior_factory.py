# ===== 全局工厂 =====
from typing import Dict, Type, List

from urm.config import Config
from urm.reward.trajectory.behavior import Behavior
from .constant import BehaviorName


class BehaviorFactory:
    _registry: Dict[str, Type[Behavior]] = {}

    def __init__(self, config: Config.RewardConfig.BehaviorConfigs):
        self.config = config

    @classmethod
    def register(cls, name: str):
        """注册装饰器"""

        def decorator(klass: Type[Behavior]):
            cls._registry[name] = klass
            klass.behavior_type = BehaviorName.from_str(name)
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

    def get_all_behaviors_by_config(self):
        """
        根据 BehaviorConfigs.behavior_configs 列表返回注册的行为类
        返回 dict: {behavior_name: class}
        """
        instances = []
        for name in getattr(self.config, "behavior_configs", []):
            if name in self._registry:
                instances.append(self._registry[name](self.config.behavior_configs))
            else:
                print(f"Warning: Behavior '{name}' 配置了但未注册")
        return instances

    def create_behavioral_combination(self, lateral_name: str, longitudinal_name: str) -> 'BehavioralCombination':
        """根据名字创建行为组合"""
        lateral_behavior = self.create(lateral_name, self.config.behavior_configs)
        longitudinal_behavior = self.create(longitudinal_name, self.config.behavior_configs)
        from .behavioral_combination import BehavioralCombination
        behavior_combination = BehavioralCombination(self.config.behavior_configs)
        behavior_combination.longitudinal = longitudinal_behavior
        behavior_combination.lateral = lateral_behavior
        return behavior_combination

    def get_all_scenarios_as_combinations(self) -> Dict[str, List['BehavioralCombination']]:
        """
        根据 config.scenarios 返回所有场景的行为组合序列
        返回: {scenario_name: [BehavioralCombination, ...]}
        """
        result = {}
        for scenario in self.config.scenarios:
            combinations = []
            for step in scenario.sequence:
                combo = self.create_behavioral_combination(step.lateral, step.longitudinal)
                combinations.append(combo)
            result[scenario.name] = combinations
        return result
