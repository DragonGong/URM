from typing import Union, Tuple, List
import numpy as np
from .region_2d import Region2D

class PolygonRegion(Region2D):
    """任意多边形区域（支持凹凸），顶点按顺序给出"""

    def __init__(self, vertices: Union[List[Tuple[float, float]], np.ndarray]):
        """
        vertices: [(x1,y1), (x2,y2), ...] 至少3个点，按顺时针或逆时针排列
        """
        if len(vertices) < 3:
            raise ValueError("多边形至少需要3个顶点")
        self._vertices = np.array(vertices, dtype=float)
        if self._vertices.shape[1] != 2:
            raise ValueError("每个顶点必须是 (x, y)")

    def contains(self, point: Union[Tuple[float, float], np.ndarray]) -> bool:
        """射线法判断点是否在多边形内"""
        x, y = point if isinstance(point, tuple) else (point[0], point[1])
        n = len(self._vertices)
        inside = False
        p1x, p1y = self._vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self._vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @property
    def center(self) -> Tuple[float, float]:
        """简单质心（算术平均）"""
        mean = np.mean(self._vertices, axis=0)
        return float(mean[0]), float(mean[1])

    @property
    def area(self) -> float:
        """鞋带公式计算面积"""
        x = self._vertices[:, 0]
        y = self._vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        min_x, min_y = np.min(self._vertices, axis=0)
        max_x, max_y = np.max(self._vertices, axis=0)
        return float(min_x), float(min_y), float(max_x), float(max_y)

    def __repr__(self):
        return f"PolygonRegion(vertices={len(self._vertices)}个)"