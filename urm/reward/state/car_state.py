import math
from copy import deepcopy
from typing import List, Tuple

from urm.reward.state.state import State
from urm.reward.state.utils.position import Position
from urm.reward.state.utils.vehicle_size import VehicleSize
from urm.reward.state.utils.velocity import Velocity


class CarState(State):
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, **kwargs):
        super().__init__(**kwargs)
        # 内部使用封装类
        self._position = Position(x, y)
        self._velocity = Velocity(vx, vy)

        self._vehicle_size = VehicleSize()

    # ========== 保持外部接口不变 ==========

    @property
    def vehicle_size(self) -> VehicleSize:
        return self._vehicle_size

    @property
    def velocity(self):
        return deepcopy(self._velocity)

    @property
    def x(self):
        return self._position.x

    @x.setter
    def x(self, value):
        self._position.x = value

    @property
    def y(self):
        return self._position.y

    @y.setter
    def y(self, value):
        self._position.y = value

    @property
    def vx(self):
        return self._velocity.vx

    @vx.setter
    def vx(self, value):
        self._velocity.vx = value

    @property
    def vy(self):
        return self._velocity.vy

    @vy.setter
    def vy(self, value):
        self._velocity.vy = value

    # ========== 方法保持行为一致 ==========

    def __repr__(self):
        return f"CarState(x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy})"

    def update_position(self, dt=1.0):
        """根据当前速度更新位置"""
        self._position.x += self._velocity.vx * dt
        self._position.y += self._velocity.vy * dt

    @property
    def speed(self):
        """返回合速度大小"""
        return self._velocity.magnitude

    @property
    def position_cls(self):
        return self._position

    @property
    def position(self):
        """返回当前位置元组 (x, y)"""
        return self._position.xy

    def set_position(self, x, y):
        self._position.x = x
        self._position.y = y

    def set_velocity(self, vx, vy):
        self._velocity.vx = vx
        self._velocity.vy = vy

    @classmethod
    def from_position(cls, position: Position):
        return cls(position.x, position.y)

    def get_bounding_box_corners(self) -> List[Tuple[float, float]]:
        """
        计算车辆当前时刻的矩形包围盒四个角的世界坐标。
        顺序：前右 -> 前左 -> 后左 -> 后右  顺时针

        Returns:
            List[Tuple[float, float]]: 四个角的 (x, y) 坐标列表
        """
        x = self._position.x
        y = self._position.y
        vx = self.velocity.vx
        vy = self.velocity.vy
        L = self._vehicle_size.length
        W = self._vehicle_size.width

        # 计算航向角（基于速度方向）
        yaw = math.atan2(vy, vx)

        # 车体坐标系下的四个角（中心为原点，x轴向前）
        front_right = (L / 2, W / 2)
        front_left = (L / 2, -W / 2)
        rear_left = (-L / 2, -W / 2)
        rear_right = (-L / 2, W / 2)
        corners = [front_right, front_left, rear_left, rear_right]

        # 旋转 + 平移 到世界坐标
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        world_corners = []
        for cx, cy in corners:
            wx = x + cos_yaw * cx - sin_yaw * cy
            wy = y + sin_yaw * cx + cos_yaw * cy
            world_corners.append((wx, wy))

        return world_corners
