from typing import Tuple, Union
import numpy as np
from .region_2d import Region2D


class RectRegion(Region2D):
    """矩形区域，由左下角和右上角定义"""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("矩形区域定义无效：x_min < x_max 且 y_min < y_max")
        self._x_min = x_min
        self._y_min = y_min
        self._x_max = x_max
        self._y_max = y_max

    def contains(self, point: Union[Tuple[float, float], np.ndarray]) -> bool:
        x, y = point if isinstance(point, tuple) else (point[0], point[1])
        return (self._x_min <= x <= self._x_max) and (self._y_min <= y <= self._y_max)

    @property
    def center(self) -> Tuple[float, float]:
        return (self._x_min + self._x_max) / 2, (self._y_min + self._y_max) / 2

    @property
    def area(self) -> float:
        return (self._x_max - self._x_min) * (self._y_max - self._y_min)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._x_min, self._y_min, self._x_max, self._y_max

    def __repr__(self):
        return f"RectRegion({self._x_min}, {self._y_min}, {self._x_max}, {self._y_max})"
