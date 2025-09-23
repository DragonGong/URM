from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from urm.config import Config
from urm.reward.state.interface import EnvInterface
from urm.reward.trajectory.traj import TrajEdge, TrajNode


class Fitting(ABC):
    def __init__(self, config: Config.RewardConfig.FittingModelConfigs, **kwargs):
        self.config = config
        self.interval_duration: float = config.interval_duration

    @abstractmethod
    def fit_edge_by_node(self, edge: TrajEdge) -> List[TrajNode]:
        ...

    @abstractmethod
    def fit_edge_by_node_frenet(self, edge: TrajEdge, env: EnvInterface) -> List[TrajNode]:
        ...


@dataclass
class ModelName:
    Polynomial = "polynomial"
