from typing import Tuple, Optional

import numpy as np

from urm.reward.state.interface import EnvInterface


class Position:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
        self.calculated_frenet = False
        self.longitudinal: float = 0.0
        self.lateral: float = 0.0
        self.lane_id: Optional[Tuple[str, str, int]] = None

    def calculate_frenet(self, env: EnvInterface):
        start, end, lane_id = env.get_current_road_segment(self.x, self.y,return_lane_id=True)
        longitudinal, lateral = env.world_to_lane_local(self.x, self.y, lane_id)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.lane_id = lane_id
        self.calculated_frenet = True

    @property
    def xy(self) -> Tuple[float, float]:
        """返回 (x, y) 元组，方便传参或解包"""
        return self.x, self.y

    def distance_to(self, other: 'Position') -> float:
        """计算到另一个 Position 的欧几里得距离"""
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx ** 2 + dy ** 2) ** 0.5

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y})"

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f})"

    def __eq__(self, other) -> bool:
        """支持 == 比较"""
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y

    def __add__(self, other: 'Position') -> 'Position':
        """支持位置相加（向量加法）"""
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Position') -> 'Position':
        """支持位置相减（向量减法）"""
        return Position(self.x - other.x, self.y - other.y)

    def copy(self) -> 'Position':
        """返回一个副本，避免引用修改"""
        return Position(self.x, self.y)

    @classmethod
    def from_tuple(cls, pos: Tuple[float, float]) -> 'Position':
        """从 (x, y) 元组构造 Position"""
        return cls(x=pos[0], y=pos[1])

    @classmethod
    def from_nparray(cls, np_arr: np.array) -> 'Position':
        return cls(x=np_arr[0], y=np_arr[1])
