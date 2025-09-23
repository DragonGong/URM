import math
from copy import deepcopy
from typing import List, Tuple

from urm.reward.state.state import State
from urm.reward.state.utils.position import Position
from urm.reward.state.utils.vehicle_size import VehicleSize
from urm.reward.state.utils.velocity import Velocity


class CarState(State):
    def __init__(self, env, x=0.0, y=0.0, vx=0.0, vy=0.0, **kwargs):
        super().__init__(env, **kwargs)
        # 内部使用封装类
        self._position = Position(x, y)
        self._position.calculate_frenet(self.env_condition)
        self._velocity = Velocity(vx, vy)

        self._vehicle_size = VehicleSize()

    # ========== 保持外部接口不变 ==========

    def set_frenet_velocity(self):
        self._velocity.set_frenet(self.env_condition, self._position.x, self._position.y)

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

    @property
    def longitudinal(self):
        assert self._position.calculated_frenet, "frenet is not calculated"
        return self._position.longitudinal

    @property
    def lateral(self):
        assert self._position.calculated_frenet, "frenet is not calculated"
        return self._position.lateral

    @property
    def lane_id(self):
        assert self._position.calculated_frenet, "frenet is not calculated"
        return self._position.lane_id

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

    def __deepcopy__(self, memo):
        # 避免重复复制
        if id(self) in memo:
            return memo[id(self)]

        # 创建新实例（不调用 __init__，避免重新绑定 env 或计算 frenet）
        cls = self.__class__
        new_obj = cls.__new__(cls)

        # 注册到 memo，防止递归复制
        memo[id(self)] = new_obj

        # 浅拷贝 env（保持引用）
        new_obj.env_condition = self.env_condition

        # 深拷贝内部状态对象
        new_obj._position = deepcopy(self._position, memo)
        new_obj._velocity = deepcopy(self._velocity, memo)
        new_obj._vehicle_size = deepcopy(self._vehicle_size, memo)

        # 如果父类 State 有其他属性，也复制（可选）
        # 注意：super().__init__ 不在这里调用！
        # 如果 State 有其他属性，手动复制：
        for k, v in self.__dict__.items():
            if k not in ['env_condition', '_position', '_velocity', '_vehicle_size']:
                setattr(new_obj, k, deepcopy(v, memo))

        return new_obj