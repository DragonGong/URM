from abc import ABC, abstractmethod

from urm.config import Config
from urm.reward.state.region2d import Region2D, CircleRegion
from urm.reward.state.utils.position import Position


class Model(ABC):
    def __init__(self, config: Config, **kwargs):
        self.config = config

    @abstractmethod
    def predict_position(self, car_state, time: float) -> Position:
        """
        根据当前状态，预测 time 秒后的位置（匀速直线运动）
        """
        ...

    @abstractmethod
    def predict_region(self, car_state, time: float, width: float = 0.0, length: float = 0.0,
                       radius: float = None) -> 'Region2D':
        """
        预测 time 秒后车辆占据的区域。
        如果传入 radius，返回圆形区域；否则返回矩形区域（AABB）。
        """
        ...

    @abstractmethod
    def predict_region_with_uncertainty(
            self,
            car_state,
            time: float,
            sigma_position: float = 0.0,
            sigma_velocity: float = 0.0,
            confidence: float = 0.95
    ) -> 'CircleRegion':
        """
        考虑不确定性，预测车辆在 time 秒后可能占据的圆形区域
        """
        ...
