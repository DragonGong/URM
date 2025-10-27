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

    @classmethod
    def from_center_and_size(cls, center_x: float, center_y: float, width: float, height: float) -> "RectRegion":
        """
        通过中心点和宽高创建矩形区域。

        参数:
            center_x (float): 矩形中心的 x 坐标
            center_y (float): 矩形中心的 y 坐标
            width (float): 矩形宽度（沿 x 轴）
            height (float): 矩形高度（沿 y 轴）

        返回:
            RectRegion: 新建的矩形区域实例
        """
        if width <= 0 or height <= 0:
            raise ValueError("矩形的宽和高必须为正数")
        half_w = width / 2.0
        half_h = height / 2.0
        x_min = center_x - half_w
        y_min = center_y - half_h
        x_max = center_x + half_w
        y_max = center_y + half_h
        return cls(x_min, y_min, x_max, y_max)

    def intersects(self, other: "RectRegion") -> bool:
        """
        判断当前矩形与另一个 RectRegion 是否相交（包括边界接触）。

        参数:
            other (RectRegion): 另一个矩形区域

        返回:
            bool: 如果相交（或接触）返回 True，否则返回 False
        """
        if not isinstance(other, RectRegion):
            raise TypeError("只能与另一个 RectRegion 实例判断相交")

        # AABB 相交条件：x 轴重叠 且 y 轴重叠
        x_overlap = not (self._x_max < other._x_min or other._x_max < self._x_min)
        y_overlap = not (self._y_max < other._y_min or other._y_max < self._y_min)
        return x_overlap and y_overlap

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
