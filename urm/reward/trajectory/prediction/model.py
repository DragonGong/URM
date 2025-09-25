from abc import ABC, abstractmethod
from dataclasses import dataclass

from urm.config import Config
from urm.reward.state.region2d import Region2D, CircleRegion
from urm.reward.state.utils.position import Position


class Model(ABC):
    def __init__(self, config: Config.RewardConfig.PredictionModelConfigs, **kwargs):
        self.config = config

    @abstractmethod
    def predict_region(self, car_state, time: float, width: float = 0.0, length: float = 0.0,
                       radius: float = None) -> 'Region2D':
        """
        预测 time 秒后车辆占据的区域。
        如果传入 radius，返回圆形区域；否则返回矩形区域（AABB）。
        """
        ...


@dataclass
class ModelName:
    LINEAR_MODEL = "linear_model"
    IDM_MODEL = "idm_model"
