from abc import ABC, abstractmethod

from urm.config import Config
from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from urm.reward.trajectory.highway_env_state import HighwayState as State


class Behavior(ABC):

    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        self.config = config

    @abstractmethod
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass
