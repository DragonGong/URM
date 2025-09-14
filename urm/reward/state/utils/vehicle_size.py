

from typing import Tuple
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=False)  # 设为 True 可使对象不可变
class VehicleSize:
    """
    表示车辆的物理尺寸：长、宽、高（单位：米）
    默认值参考标准乘用车,这个默认值和highway-env是一样的
    """
    length: float = 5.0   # 车长（沿行驶方向）
    width: float = 2.0    # 车宽（横向）
    height: float = 1.5   # 车高（可选，用于3D或可视化）

    def __post_init__(self):
        if self.length <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("车辆尺寸必须为正数")

    @property
    def lw(self) -> Tuple[float, float]:
        """返回 (长, 宽) 元组，常用于2D碰撞检测"""
        return self.length, self.width

    @property
    def lwh(self) -> Tuple[float, float, float]:
        """返回 (长, 宽, 高) 元组"""
        return self.length, self.width, self.height

    def scale(self, factor: float) -> 'VehicleSize':
        """
        按比例缩放车辆尺寸（例如用于不同车型）
        返回新对象（不修改原对象）
        """
        return VehicleSize(
            length=self.length * factor,
            width=self.width * factor,
            height=self.height * factor
        )

    def __repr__(self):
        return f"VehicleSize(L={self.length:.1f}m, W={self.width:.1f}m, H={self.height:.1f}m)"

    def __eq__(self, other):
        if not isinstance(other, VehicleSize):
            return False
        return (
            abs(self.length - other.length) < 1e-6 and
            abs(self.width - other.width) < 1e-6 and
            abs(self.height - other.height) < 1e-6
        )

    def copy(self) -> 'VehicleSize':
        """返回深拷贝"""
        return deepcopy(self)
