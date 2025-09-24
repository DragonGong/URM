import math
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

        vehicle_size = root.car_state.vehicle_size
        x_min, x_max = min(xs) - vehicle_size.length, max(xs) + vehicle_size.length
        y_min, y_max = min(ys) - vehicle_size.width, max(ys) + vehicle_size.width

        assert self.step_num != 0, "step_num is 0"
        self.maps = [RiskMap(x_min, x_max, y_min, y_max, self.cell_size)
                     for _ in range(self.step_num)]

    def world_to_local(self, x, y):
        delta = np.array([x, y]) - self.root_pos
        return np.dot(delta, self.e_x), np.dot(delta, self.e_y)

    def assign_risk(self):
        """
        将 trajtree上的点映射到riskmap上
        """
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

    def assign_risk_with_vehicle(self):
        for node in self.trajtree.get_all_nodes_with_edge_nodes():
            if node.risk is None:
                continue
            t = get_interval_index(node.get_time(), self.duration, self.step_num)
            if 0 <= t < self.step_num:
                risk_value = node.risk.get_value()
                if risk_value is not None:
                    corners = node.car_state.get_bounding_box_corners()
                    local_corners = []
                    for point in corners:
                        local_corners.append(self.world_to_local(*point))
                    self.maps[t].add_rectangle(local_corners, risk_value)

    def plot_all(self):
        """
        将所有的RiskMap绘制在一个大窗口中，形成多个小图的情景。
        """
        # 根据maps的数量决定subplot的布局
        n = len(self.maps)
        cols = math.ceil(math.sqrt(n))  # 列数
        rows = math.ceil(n / cols)  # 行数

        # 创建大窗口和子图
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.ravel() if n > 1 else [axes]  # 如果只有一个子图，确保axes是一个列表

        for idx, risk_map in enumerate(self.maps):
            ax = axes[idx]
            risk_avg = risk_map.finalize()
            mask = (risk_map.count > 0)

            # 创建绿 -> 红的 colormap
            cmap = LinearSegmentedColormap.from_list(
                'green_to_red',
                [
                    (0.0, (0, 1, 0)),  # Green
                    (0.25, (1, 1, 0)),  # Yellow
                    (0.5, (1, 0.647, 0)),  # Orange
                    (0.75, (1, 0.27, 0)),  # Orange-red
                    (1.0, (1, 0, 0))  # Red
                ],
                N=256
            )

            # 将未覆盖区域（count=0）设为 NaN
            risk_display = np.where(mask, risk_avg, np.nan)

            extent = (float(risk_map.x_min), float(risk_map.x_max), float(risk_map.y_min), float(risk_map.y_max))

            # 绘制图像
            cax = ax.imshow(risk_display, origin='lower', extent=extent, cmap=cmap,
                            vmin=0.0, vmax=1.0, interpolation='nearest', aspect='auto')

            ax.set_title(f"Time step {idx}")
            ax.set_xlabel("local x (m)")
            ax.set_ylabel("local y (m)")
            ax.set_aspect('equal', 'box')  # 保持纵横比

            # 添加colorbar到每个子图
            plt.colorbar(cax, ax=ax, label="Risk Level (0=Green, 1=Red)")

        # 隐藏多余的子图
        for idx in range(n, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show(block=True)

    def sum_all(self) -> 'RiskMap':
        """
        将所有时间片的 RiskMap 累加，返回一个新的 RiskMap 对象
        risk_sum 和 count 逐元素相加
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

    def _get_occupied_mask_from_nodes(self, traj_nodes, risk_map: 'RiskMap') -> np.ndarray:
        """
        根据 traj_nodes 中的节点位置，生成 risk_map 上被占据的格子的布尔 mask。

        Parameters:
            traj_nodes: List[TrajNode]
            risk_map: RiskMap 实例

        Returns:
            mask: np.ndarray[bool]，shape == (ny, nx)，True 表示该格子被至少一个 node 覆盖
        """
        mask = np.zeros((risk_map.ny, risk_map.nx), dtype=bool)

        for node in traj_nodes:
            # 转换到局部坐标系
            x_local, y_local = self.world_to_local(node.x, node.y)

            # 计算对应的网格索引
            i = int((x_local - risk_map.x_min) / risk_map.cell_size)
            j = int((y_local - risk_map.y_min) / risk_map.cell_size)

            # 边界检查
            if 0 <= i < risk_map.nx and 0 <= j < risk_map.ny:
                mask[j, i] = True

        return mask

    def get_risk_by_tree(self, traj_nodes, risk_map: 'RiskMap') -> (float, int, RiskMap):
        """
        计算 traj_nodes 覆盖的 risk_map 格子的总风险值和格子数量。

        注意：多个节点落在同一格子只计一次（通过 mask 去重）。

        Returns:
            total_risk (float): 所有被覆盖格子的平均风险值之和（即 Σ(risk_avg)）
            num_cells (int): 被覆盖的格子数量
        """
        # 1. 生成被占据的格子 mask
        mask = self._get_occupied_mask_from_nodes(traj_nodes, risk_map)

        # 2. 计算每个格子的平均风险（避免除零）
        risk_avg = np.zeros_like(risk_map.risk_sum)
        valid = risk_map.count > 0
        risk_avg[valid] = risk_map.risk_sum[valid] / risk_map.count[valid]

        # 3. 只对 mask 为 True 且 count > 0 的格子求和（若 count==0，risk_avg=0，不影响）
        total_risk = float(np.sum(risk_avg[mask]))
        num_cells = int(np.sum(mask))
        riskmap_mask = self.mask_riskmap(risk_map, mask)
        return total_risk, num_cells, riskmap_mask

    def mask_riskmap(self, risk_map: 'RiskMap', mask: np.ndarray) -> 'RiskMap':
        """
        根据给定的布尔 mask，生成一个新的 RiskMap：
        - 仅保留 mask 为 True 的格子的 risk_sum 和 count；
        - 其余格子的 risk_sum 和 count 设为 0。

        Parameters:
            risk_map: 原始 RiskMap 对象
            mask: 布尔数组，shape 必须为 (ny, nx)，与 risk_map 的网格一致

        Returns:
            新的 RiskMap 对象
        """
        # 检查 mask 形状是否匹配
        if mask.shape != (risk_map.ny, risk_map.nx):
            raise ValueError(
                f"Mask shape {mask.shape} 不匹配 RiskMap 网格 {(risk_map.ny, risk_map.nx)}"
            )

        # 创建新的 RiskMap（复用原图的边界和 cell_size）
        new_map = RiskMap(
            x_min=risk_map.x_min,
            x_max=risk_map.x_max,
            y_min=risk_map.y_min,
            y_max=risk_map.y_max,
            cell_size=risk_map.cell_size
        )

        # 应用 mask：只保留 mask 为 True 的区域
        new_map.risk_sum = np.where(mask, risk_map.risk_sum, 0.0)
        new_map.count = np.where(mask, risk_map.count, 0)

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
        if index > total_intervals:
            raise ValueError(f"时间 {time} 超出最大范围（共 {total_intervals} 个时间段）")
        if index == total_intervals:
            index = total_intervals - 1

    return index
