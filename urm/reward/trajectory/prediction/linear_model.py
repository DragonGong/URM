from urm.config import Config
from urm.reward.trajectory.prediction.model import Model

from urm.reward.trajectory.prediction.model import Model
from urm.reward.state.utils.position import Position
import numpy as np
from urm.reward.state.region2d import RectRegion, CircleRegion, Region2D


class LinearModel(Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)

    def predict_position(self, car_state, time: float) -> Position:
        """
        根据当前状态，预测 time 秒后的位置（匀速直线运动）
        """
        x = car_state.x + car_state.vx * time
        y = car_state.y + car_state.vy * time
        return Position(x, y)

    def predict_region(self, car_state, time: float, width: float = 0.0, length: float = 0.0,
                       radius: float = None) -> 'Region2D':
        """
        预测 time 秒后车辆占据的区域。
        如果传入 radius，返回圆形区域；否则返回矩形区域（AABB）。
        """

        future_pos = self.predict_position(car_state, time)
        cx, cy = future_pos.x, future_pos.y

        if radius is not None:
            return CircleRegion(cx, cy, radius)
        else:
            half = max(width, length) / 2
            return RectRegion(cx - half, cy - half, cx + half, cy + half)

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

        future_pos = self.predict_position(car_state, time)
        cx, cy = future_pos.x, future_pos.y

        total_sigma = (sigma_position ** 2 + (sigma_velocity * time) ** 2) ** 0.5

        # 置信系数近似
        k = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }.get(round(confidence, 2), 2.0)

        radius = k * total_sigma
        return CircleRegion(cx, cy, radius)
