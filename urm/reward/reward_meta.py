from abc import abstractmethod, ABC

from urm.config import Config


class RewardMeta(ABC):
    def __init__(self, config: Config, **kwargs):
        self.config = config

    @abstractmethod
    def reward(self):
