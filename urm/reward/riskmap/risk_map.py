import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Polygon, box
from urm.reward.state.car_state import CarState
from urm.reward.trajectory.traj_tree import TrajTree

colors = [(1, 1, 1), (1, 0, 0)]  # 白色 -> 红色
n_bins = 256  # 色彩等级数
cmap_name = 'white_to_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


class RiskMap:
    def __init__(self, x_min, x_max, y_min, y_max, cell_size):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.cell_size = cell_size
        self.nx = math.ceil((x_max - x_min) / cell_size)
        self.ny = math.ceil((y_max - y_min) / cell_size)

        self.risk_sum = np.zeros((self.ny, self.nx))
        self.count = np.zeros((self.ny, self.nx), dtype=int)

    def add_point(self, x, y, risk_value):
        i = int((x - self.x_min) / self.cell_size)
        j = int((y - self.y_min) / self.cell_size)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            self.risk_sum[j, i] += risk_value
            self.count[j, i] += 1

    def finalize(self):
        risk_avg = np.zeros_like(self.risk_sum)
        mask = self.count > 0
        risk_avg[mask] = self.risk_sum[mask] / self.count[mask]
        return risk_avg

    def plot(self, ax=None, title: str = "RiskMap", show_colorbar: bool = True, cmap='hot', interpolation='nearest'):
        risk_avg = self.finalize()
        plt.close("all")
        if ax is None:
            # plt.ion()
            fig, ax = plt.subplots(figsize=(6, 6))
            created = True
        else:
            created = False

        extent = (float(self.x_min), float(self.x_max), float(self.y_min), float(self.y_max))
        # imshow 期望 extent 为长度 4 的序列（tuple 更稳妥）
        cax = ax.imshow(risk_avg, origin='lower', extent=extent, cmap=cmap,
                        interpolation=interpolation, aspect='auto')

        ax.set_title(title)
        ax.set_xlabel("local x (m)")
        ax.set_ylabel("local y (m)")
        if show_colorbar:
            plt.colorbar(cax, ax=ax)
        if created:
            plt.tight_layout()
            plt.show(block=True)
            plt.pause(0.001)

    def plot_version_01(self, ax=None, title: str = "RiskMap", show_colorbar: bool = True, cmap=cm,
                        interpolation='nearest'):
        risk_avg = self.finalize()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created = True
        else:
            created = False

        extent = (float(self.x_min), float(self.x_max), float(self.y_min), float(self.y_max))
        cax = ax.imshow(risk_avg, origin='lower', extent=extent, cmap=cmap,
                        interpolation=interpolation, aspect='auto')

        ax.set_title(title)
        ax.set_xlabel("local x (m)")
        ax.set_ylabel("local y (m)")
        if show_colorbar:
            plt.colorbar(cax, ax=ax)
        if created:
            plt.tight_layout()
            plt.show(block=True)
        return

    def get_visualization_data(self):
        """
        返回可视化所需数据，不进行绘图
        返回字典:
        {
            'risk_avg': np.ndarray,
            'extent': (x_min, x_max, y_min, y_max)
        }
        """
        risk_avg = self.finalize()
        extent = (float(self.x_min), float(self.x_max), float(self.y_min), float(self.y_max))
        return {'risk_avg': risk_avg, 'extent': extent}

    def get_risk_for_car(self, car: 'CarState', world_to_local) -> float:
        """
        给定一个 CarState（含世界坐标），计算它在 riskmap 上覆盖区域的平均风险。
        :param car: CarState 对象
        :param world_to_local: 函数 (x_world,y_world)->(x_local,y_local)，由 RiskMapManager 提供
        :return: 平均风险值 (float)，如果覆盖区域没有网格则返回 0
        """
        # 车中心转到局部坐标系
        cx, cy = world_to_local(car.x, car.y)

        # 朝向由速度方向决定（如果速度为零，则默认朝向 x 轴）
        vx, vy = car.vx, car.vy
        if abs(vx) + abs(vy) < 1e-6:
            heading = 0.0
        else:
            heading = np.arctan2(vy, vx)

        # 车长宽
        L, W = car.vehicle_size.length, car.vehicle_size.width

        # 在车体局部坐标系下的矩形四角（中心为原点，车头朝 +x）
        local_corners = np.array([
            [L / 2, W / 2],
            [L / 2, -W / 2],
            [-L / 2, -W / 2],
            [-L / 2, W / 2]
        ])

        # 旋转到局部坐标系（旋转 heading，再平移到 cx,cy）
        rot = np.array([[np.cos(heading), -np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])
        rotated = local_corners @ rot.T + np.array([cx, cy])

        car_poly = Polygon(rotated)

        # 遍历所有网格，检查相交
        risks = []
        for j in range(self.ny):
            for i in range(self.nx):
                if self.count[j, i] == 0:
                    continue
                # 网格中心
                x0 = self.x_min + i * self.cell_size
                y0 = self.y_min + j * self.cell_size
                # 网格边界（矩形）
                cell_poly = box(x0, y0, x0 + self.cell_size, y0 + self.cell_size)
                if car_poly.intersects(cell_poly):
                    risk_val = self.risk_sum[j, i] / self.count[j, i]
                    risks.append(risk_val)

        if risks:
            return float(np.mean(risks))
        else:
            return 0.0
