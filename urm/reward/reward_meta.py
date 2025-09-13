from abc import abstractmethod, ABC

from urm.config import Config
from urm.reward.state.ego_state import EgoState
from urm.reward.state.interface import EnvInterface
from urm.reward.state.surrounding_state import SurroundingState


class RewardMeta(ABC):
    def __init__(self, config: Config, **kwargs):
        self.config = config

    @abstractmethod
    def reward(self, ego_state: EgoState, surrounding_states: SurroundingState, env_condition: EnvInterface,
               baseline_reward):
        ...
