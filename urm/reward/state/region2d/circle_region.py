from typing import Tuple, Union
import numpy as np
from .region_2d import Region2D

class CircleRegion(Region2D):
    """圆形区域，由圆心和半径定义"""

    def __init__(self, center_x: float, center_y: float, radius: float):
        if radius <= 0:
            raise ValueError("半径必须大于 0")
        self._center_x = center_x
        self._center_y = center_y
        self._radius = radius

    def contains(self, point: Union[Tuple[float, float], np.ndarray]) -> bool:
        x, y = point if isinstance(point, tuple) else (point[0], point[1])
        dx = x - self._center_x
        dy = y - self._center_y
        return dx * dx + dy * dy <= self._radius * self._radius

    @property
    def center(self) -> Tuple[float, float]:
        return self._center_x, self._center_y

    @property
    def area(self) -> float:
        return 3.141592653589793 * self._radius ** 2

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        r = self._radius
        return self._center_x - r, self._center_y - r, self._center_x + r, self._center_y + r

    def __repr__(self):
        return f"CircleRegion(center=({self._center_x}, {self._center_y}), radius={self._radius})"