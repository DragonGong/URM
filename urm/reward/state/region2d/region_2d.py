from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np


class Region2D(ABC):
    """2D 区域的抽象基类 —— 所有具体区域类必须继承并实现以下接口"""

    @abstractmethod
    def contains(self, point: Union[Tuple[float, float], np.ndarray]) -> bool:
        """判断点是否在区域内"""
        pass

    @property
    @abstractmethod
    def center(self) -> Tuple[float, float]:
        """返回区域中心坐标 (x, y)"""
        pass

    @property
    @abstractmethod
    def area(self) -> float:
        """返回区域面积"""
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[float, float, float, float]:
        """返回边界 (x_min, y_min, x_max, y_max)"""
        pass