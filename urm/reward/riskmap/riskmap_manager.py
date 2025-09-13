import math
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from urm.config import Config
from urm.reward.riskmap.risk_map import RiskMap
from urm.reward.trajectory.traj_tree import TrajTree


class RiskMapManager:
    def __init__(self, config: Config.RewardConfig, trajtree: TrajTree):
        self.trajtree = trajtree
        self.step_num = config.step_num
        self.cell_size = config.riskmap_config.cell_size
        self.duration = config.duration
        # root info
        root = trajtree.root
        v = root.velocity
        self.root_pos = np.array([root.x, root.y])
        self.e_x = np.array([v.vx, v.vy]) / v.magnitude
        self.e_y = np.array([-v.vy, v.vx]) / v.magnitude

        # 确定 bounding box
        all_nodes = trajtree.get_all_nodes()
        coords = [self.world_to_local(n.x, n.y) for n in all_nodes]
        xs, ys = zip(*coords)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        assert self.step_num != 0, "step_num is 0"
        self.maps = [RiskMap(x_min, x_max, y_min, y_max, self.cell_size)
                     for _ in range(self.step_num)]

    def world_to_local(self, x, y):
        delta = np.array([x, y]) - self.root_pos
        return np.dot(delta, self.e_x), np.dot(delta, self.e_y)

    def assign_risk(self):
        # 遍历所有节点，按时间步分类
        for node in self.trajtree.get_all_nodes_with_edge_nodes():
            if node.risk is None:
                continue
            t = get_interval_index(node.get_time(), self.duration, self.step_num)  # 得到所对应的这个时间段
            if 0 <= t < self.step_num:
                risk_value = node.risk.get_value()
                if risk_value is not None:
                    x, y = self.world_to_local(node.x, node.y)
                    self.maps[t].add_point(x, y, risk_value)

    def plot_all(self, figsize_per_map: Tuple[float, float] = (6, 6), cmap='hot'):
        n = self.step_num
        fig, axes = plt.subplots(1, n, figsize=(figsize_per_map[0] * n, figsize_per_map[1]))
        if n == 1:
            axes = [axes]
        for i, (ax, rm) in enumerate(zip(axes, self.maps)):
            rm.plot(ax=ax,
                    title=f"RiskMap t_idx={i} (t ∈ [{i * self.duration:.2f},{(i + 1) * self.duration:.2f})s)",
                    show_colorbar=True, cmap=cmap)
        plt.tight_layout()
        plt.show()

    def sum_all(self) -> 'RiskMap':
        """
        将所有时间片的 RiskMap 累加，返回一个新的 RiskMap 对象
        （risk_sum 和 count 逐元素相加，最后可再 finalize 或直接 plot）。
        """
        if not self.maps:
            raise ValueError("没有风险图可累加")

        # 复制第一个 map 的范围与 cell_size
        first = self.maps[0]
        new_map = RiskMap(first.x_min, first.x_max, first.y_min, first.y_max, first.cell_size)

        # 累加所有 maps
        for rm in self.maps:
            new_map.risk_sum += rm.risk_sum
            new_map.count += rm.count

        return new_map


def get_interval_index(time: float, duration: float, total_intervals: int = None) -> int:
    """
    根据时间点，返回其所属的时间段索引（从 0 开始）

    :param time: 查询的时间点（float）
    :param duration: 每个时间段的长度（float）
    :param total_intervals: 总时间段数（可选，用于边界检查）
    :return: 时间段索引（int）
    :raises ValueError: 如果超出范围（当 total_intervals 提供时）
    """
    if duration <= 0:
        raise ValueError("duration 必须大于 0")

    index = int(time // duration)  # 使用 floor 除法

    if total_intervals is not None:
        if index < 0:
            raise ValueError(f"时间 {time} 小于 0，不在有效范围内")
        if index >= total_intervals:
            raise ValueError(f"时间 {time} 超出最大范围（共 {total_intervals} 个时间段）")

    return index
