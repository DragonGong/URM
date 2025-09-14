import math
from typing import Tuple


class Velocity:
    def __init__(self, vx: float = 0.0, vy: float = 0.0):
        self.vx = vx
        self.vy = vy

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.vx, self.vy)

    @property
    def direction(self) -> float:
        """返回速度矢量的朝向（弧度），范围 [-π, π]，相对于 x 轴正方向"""
        return math.atan2(self.vy, self.vx)

    @property
    def magnitude(self) -> float:
        """速度大小（标量速度）"""
        return (self.vx ** 2 + self.vy ** 2) ** 0.5

    def __repr__(self) -> str:
        return f"Velocity(vx={self.vx}, vy={self.vy})"

    def __str__(self) -> str:
        return f"({self.vx:.2f}, {self.vy:.2f})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Velocity):
            return False
        return self.vx == other.vx and self.vy == other.vy

    def __add__(self, other: 'Velocity') -> 'Velocity':
        return Velocity(self.vx + other.vx, self.vy + other.vy)

    def __sub__(self, other: 'Velocity') -> 'Velocity':
        return Velocity(self.vx - other.vx, self.vy - other.vy)

    def copy(self) -> 'Velocity':
        return Velocity(self.vx, self.vy)

    @classmethod
    def from_tuple(cls, vel: Tuple[float, float]) -> 'Velocity':
        return cls(vx=vel[0], vy=vel[1])
