import math
from typing import List, Tuple
import copy

from urm.config import Config
from urm.reward.state.idm_state import IDMState
from urm.reward.state.car_state import CarState
from urm.reward.state.region2d import Region2D, CircleRegion, RectRegion
from .model import Model, ModelName
import numpy as np
from .model_factory import register_model


@register_model(ModelName.IDM_MODEL)
class IDMModel(Model):
    """
    基于 IDM (Intelligent Driver Model) 的轨迹预测模型。
    """

    def __init__(self, config:Config.RewardConfig.PredictionModelConfigs, **kwargs):
        super().__init__(config, **kwargs)

    def predict_region(self, car_state: IDMState, time: float, width: float = 0.0, length: float = 0.0,
                       radius: float = None) -> 'Region2D':
        original_vehicle = self._find_original_vehicle(car_state)
        cloned_vehicle = copy.deepcopy(original_vehicle)
        cloned_vehicle.road = original_vehicle.road  # 共享 road 网络
        dt = car_state.env_condition.get_env().unwrapped.config.get("simulation_frequency", 15.0)
        steps = int(time * dt)
        for _ in range(steps):
            cloned_vehicle.act()  # 决策（变道 + 加速）
            cloned_vehicle.step(dt=1.0 / dt)  # 物理更新
        final_x, final_y = cloned_vehicle.position
        final_heading = cloned_vehicle.heading
        if radius is not None:
            return CircleRegion(final_x, final_y, radius)
        else:
            L = length or cloned_vehicle.LENGTH
            W = width or cloned_vehicle.WIDTH
            corners = self._get_corners(final_x, final_y, L, W, final_heading)
            return create_rect_from_corners(np_to_correct_tuple_list(corners))

    def _find_original_vehicle(self, car_state: IDMState):
        return car_state.get_original_vehicle()

    def _get_corners(self, x: float, y: float, length: float, width: float, heading: float):
        l = length / 2.0
        w = width / 2.0
        corners_local = np.array([
            [l, w],
            [l, -w],
            [-l, -w],
            [-l, w]
        ])
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R = np.array([[cos_h, -sin_h],
                      [sin_h, cos_h]])
        corners_rotated = corners_local @ R.T
        corners_world = corners_rotated + np.array([x, y])

        return corners_world  # shape: (4, 2)


def create_rect_from_corners(corners_world: List[Tuple[float, float]]) -> RectRegion:
    if not corners_world or len(corners_world) < 2:
        raise ValueError("至少需要2个角点才能创建矩形区域")

    # 提取所有x和y坐标
    xs = [x for x, y in corners_world]
    ys = [y for x, y in corners_world]

    # 计算最小和最大值
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 创建并返回RectRegion实例
    return RectRegion(x_min, y_min, x_max, y_max)


def np_to_correct_tuple_list(arr: np.ndarray) -> List[Tuple[float, float]]:
    if arr.shape != (4, 2):
        raise ValueError("数组形状必须为(4,2)")
    return [(float(row[0]), float(row[1])) for row in arr]
