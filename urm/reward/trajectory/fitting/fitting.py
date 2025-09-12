from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from urm.config import Config
from urm.reward.trajectory.traj import TrajEdge, TrajNode


class Fitting(ABC):
    def __init__(self, config: Config.RewardConfig.FittingModelConfigs, **kwargs):
        self.config = config

    @abstractmethod
    def fit_edge_by_node(self, edge: TrajEdge) -> List[TrajNode]:
        ...


@dataclass
class ModelName:
    Polynomial = "polynomial"
