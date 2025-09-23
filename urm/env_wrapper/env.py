from typing import Union, Tuple, Any

import gymnasium as gym
import numpy as np

from urm.config import Config
from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from urm.reward.state.interface.env_interface import EnvInterface
import highway_env


class Env(gym.Wrapper):
    def __init__(self, env, config: Config, **kwargs):
        super().__init__(env)
        self.config = config
        print(f"get_road_width_info :{self.get_road_width_info()}")

    def _extract_xy(self, pos: Any) -> Tuple[float, float]:
        """
        从多种输入格式中提取 x, y 坐标

        支持：
        - (x, y) 元组或列表
        - Position(x, y) 对象
        - numpy array [x, y]
        - dict {'x': x, 'y': y}
        """
        if isinstance(pos, (tuple, list, np.ndarray)) and len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
        elif hasattr(pos, 'x') and hasattr(pos, 'y'):  # 支持 Position 类
            x, y = float(pos.x), float(pos.y)
        elif isinstance(pos, dict) and 'x' in pos and 'y' in pos:
            x, y = float(pos['x']), float(pos['y'])
        else:
            raise ValueError(f"无法从 {type(pos)} 类型中提取 x, y 坐标: {pos}")

        return x, y

    def judge_match_road(
            self,
            pos: Union[Tuple[float, float], np.ndarray, dict],
            margin: float = 0.5
    ) -> bool:
        """
        判断给定坐标 (x, y) 是否在道路安全范围内（未离开道路、未碰撞边界）

        Args:
            pos: 全局坐标系下的位置
            margin: 安全余量（单位：米），防止贴边误判
        Returns:
            bool: True 表示在道路上（安全），False 表示已离开或碰撞
        """
        env = self.env.unwrapped
        position = self._extract_xy(pos)
        road = env.road

        for lane in road.network.lanes_list():
            longitudinal, lateral = lane.local_coordinates(position)

            # 横向检查：是否在车道宽度内（考虑安全 margin）
            if abs(lateral) > lane.width / 2 - margin:
                continue

            # 纵向检查：是否在车道起止范围内（考虑 margin）
            # highway-env 默认车道很长，但为通用性保留此检查
            if longitudinal < -margin or longitudinal > getattr(lane, 'length', float('inf')) + margin:
                continue

            return True  # 在安全区域内

        return False  # 不在任何车道内 → 离开道路

    def judge_on_road(self, x, y, length, width, direction, margin: float = 0.2) -> bool:
        """
        判断车辆是否在大马路内（基于 lane 的 local_coordinates）,车的整个完整体都冲出马路才算在马路外。
        Args:
            vehicle: 车辆对象
            margin: 安全余量
        Returns:
            bool: True=在马路上, False=冲出马路
        """
        env = self.env.unwrapped
        road = env.road

        cx, cy = x, y
        length, width, heading = length, width, direction
        corners = self._vehicle_corners(cx, cy, length, width, heading)

        # print(f"get_road_width_info :{self.get_road_width_info()}")
        # 检查矩形四个角是否至少有一个点在任意 lane 内
        for corner in corners:
            for lane in road.network.lanes_list():
                longitudinal, lateral = lane.local_coordinates(corner)

                # 横向是否在车道范围内
                if abs(lateral) <= lane.width / 2 - margin:
                    # 纵向对无限车道不做限制
                    if getattr(lane, "length", None) is None or \
                            (0 - margin <= longitudinal <= lane.length + margin):
                        return True
        return False

    def get_road_width_info(self) -> dict:
        """
        获取道路宽度信息：
        - 每个车道的宽度
        - 每段并行道路的总宽度（按 start_node->end_node 分组）
        - 整条道路的最大宽度
        """
        env = self.env.unwrapped
        road = env.road
        network = road.network

        lane_widths = []
        segment_groups = {}

        for start_node, inner_dict in network.graph.items():
            for end_node, lanes_list in inner_dict.items():  # ← lanes_list 是 list
                for lane in lanes_list:  # ← 直接遍历 Lane 对象
                    lane_widths.append(lane.width)

                    key = (start_node, end_node)
                    if key not in segment_groups:
                        segment_groups[key] = []
                    segment_groups[key].append(lane.width)

        segment_widths = {
            f"{start}->{end}": sum(widths)
            for (start, end), widths in segment_groups.items()
        }

        max_road_width = max(segment_widths.values()) if segment_widths else 0.0

        return {
            'lane_widths': lane_widths,
            'segment_widths': segment_widths,
            'max_road_width': max_road_width
        }

    def _vehicle_corners(self, cx, cy, length, width, heading):
        """计算车辆矩形的四个角点"""
        dx = length / 2
        dy = width / 2
        corners = np.array([
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy]
        ])
        # 旋转 + 平移
        rot = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])
        return [tuple(rot @ c + np.array([cx, cy])) for c in corners]

    def world_to_lane_local(
            self,
            x: float,
            y: float,
            lane_id: Tuple[str, str, int]
    ) -> Tuple[float, float]:
        """
        将世界坐标 (x, y) 转换为指定车道 lane_id 上的局部坐标 (longitudinal, lateral)

        Args:
            x (float): 世界坐标 x
            y (float): 世界坐标 y
            lane_id (Tuple[str, str, int]): 车道索引，如 ('a', 'b', 0)

        Returns:
            Tuple[float, float]: (longitudinal, lateral) —— 沿车道距离 & 横向偏移

        Raises:
            ValueError: 如果 lane_id 不存在
        """
        env = self.env.unwrapped
        road = env.road
        network = road.network

        try:
            lane = network.get_lane(lane_id)
        except KeyError:
            raise ValueError(f"车道 {lane_id} 不存在于路网中")

        position = np.array([x, y])
        longitudinal, lateral = lane.local_coordinates(position)

        return float(longitudinal), float(lateral)

    def lane_local_to_world(
            self,
            lane_id: Tuple[str, str, int],
            longitudinal: float,
            lateral: float
    ) -> Tuple[float, float]:
        """
        将指定车道上的局部坐标 (longitudinal, lateral) 转换为世界坐标 (x, y)

        Args:
            lane_id (Tuple[str, str, int]): 车道索引，如 ('a', 'b', 0)
            longitudinal (float): 沿车道中心线的弧长距离
            lateral (float): 相对于车道中心线的横向偏移

        Returns:
            Tuple[float, float]: (x, y) —— 世界坐标

        Raises:
            ValueError: 如果 lane_id 不存在
        """
        env = self.env.unwrapped
        road = env.road
        network = road.network

        try:
            lane = network.get_lane(lane_id)
        except KeyError:
            raise ValueError(f"车道 {lane_id} 不存在于路网中")

        world_position = lane.position(longitudinal, lateral)
        return float(world_position[0]), float(world_position[1])

    def get_current_road_segment(
            self,
            x: float,
            y: float,
            return_lane_id: bool = False
    ) -> Union[Tuple[str, str], Tuple[str, str, Tuple[str, str, int]]]:

        """
        根据世界坐标 (x, y) 获取当前所在路段的起始节点和终止节点字符串

        Args:
            x (float): 世界坐标 x
            y (float): 世界坐标 y
            return_lane_id (bool): 是否同时返回对应的 lane_id

        Returns:
            如果 return_lane_id=False → (start_node: str, end_node: str)
            如果 return_lane_id=True  → (start_node: str, end_node: str, lane_id: Tuple[str, str, int])

        Raises:
            ValueError: 如果找不到任何车道（如车辆已飞出地图）
        """
        env = self.env.unwrapped
        network = env.road.network
        position = np.array([x, y])

        closest_lane = None
        min_distance = float('inf')
        best_lane_id = None

        # 遍历所有车道，找距离 (x,y) 最近的那条
        for start_node in network.graph:
            for end_node in network.graph[start_node]:
                for idx, lane in enumerate(network.graph[start_node][end_node]):
                    # 计算点到车道中心线的横向距离（绝对值）
                    longitudinal, lateral = lane.local_coordinates(position)
                    distance = abs(lateral)  # 横向距离作为“接近度”指标

                    # 可选：也可用欧氏距离到车道起点/终点/投影点，但横向距离更符合“在车道上”的语义
                    if distance < min_distance:
                        min_distance = distance
                        closest_lane = lane
                        best_lane_id = (start_node, end_node, idx)

        if closest_lane is None:
            raise ValueError(f"在位置 ({x}, {y}) 未找到任何车道 —— 车辆可能已离开道路或环境路网为空")

        start_node, end_node, _ = best_lane_id

        if return_lane_id:
            return start_node, end_node, best_lane_id
        else:
            return start_node, end_node

    def get_frenet_velocity(self, x: float, y: float, vx: float, vy: float) -> Tuple[float, float]:
        """
        根据世界坐标 (x, y) 和速度 (vx, vy)，计算当前车道坐标系下的纵向和横向速度。

        返回: (v_lon, v_lat)
            - v_lon: 沿车道方向的速度（正 = 前进方向）
            - v_lat: 垂直于车道方向的速度（正 = 向左 / 车道中心线左侧）
        """
        env = self.env.unwrapped
        network = env.road.network
        position = np.array([x, y])
        velocity = np.array([vx, vy])

        # Step 1: 找到最近的车道
        closest_lane = None
        min_lateral = float('inf')
        best_lane_id = None

        for start_node in network.graph:
            for end_node in network.graph[start_node]:
                for idx, lane in enumerate(network.graph[start_node][end_node]):
                    longitudinal, lateral = lane.local_coordinates(position)
                    if abs(lateral) < abs(min_lateral):
                        min_lateral = lateral
                        closest_lane = lane
                        best_lane_id = (start_node, end_node, idx)

        if closest_lane is None:
            raise ValueError(f"在位置 ({x}, {y}) 未找到任何车道")

        # Step 2: 计算车道在该点的切向量（单位向量，指向车道前进方向）
        # 方法：取 s 和 s+ds 两点，差分得切向量
        longitudinal, lateral = closest_lane.local_coordinates(position)
        ds = 0.1  # 微小步长
        s1 = longitudinal
        s2 = longitudinal + ds

        # 获取两个点的世界坐标（保持 d = lateral 不变，沿车道中心线微移）
        p1 = closest_lane.position(s1, lateral)
        p2 = closest_lane.position(s2, lateral)

        tangent_vector = np.array(p2) - np.array(p1)
        tangent_norm = np.linalg.norm(tangent_vector)

        if tangent_norm < 1e-6:
            # 避免除零，使用默认方向（如x轴）
            tangent_unit = np.array([1.0, 0.0])
        else:
            tangent_unit = tangent_vector / tangent_norm

        # Step 3: 计算法向量（垂直于切向，指向左侧为正）
        # 2D 旋转90度： (x, y) -> (-y, x) 是逆时针90度（左手法则，指向左侧）
        normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])

        # Step 4: 投影速度到切向和法向
        v_lon = np.dot(velocity, tangent_unit)  # 纵向速度
        v_lat = np.dot(velocity, normal_unit)  # 横向速度（正 = 向左）

        return v_lon, v_lat

    def frenet_velocity_to_cartesian(
            self,
            x: float,
            y: float,
            v_lon: float,
            v_lat: float
    ) -> tuple[np.ndarray[tuple[int, ...], Any], np.ndarray[tuple[int, ...], Any]]:
        """
        将 Frenet 坐标系下的速度 (v_lon, v_lat) 转换为笛卡尔坐标系下的速度 (vx, vy)

        Args:
            x, y: 车辆当前世界坐标（用于定位所在车道及方向）
            v_lon: 沿车道方向的速度（正 = 前进）
            v_lat: 垂直于车道方向的速度（正 = 向左）

        Returns:
            (vx, vy): 世界坐标系下的速度矢量
        """
        env = self.env.unwrapped
        network = env.road.network
        position = np.array([x, y])

        # Step 1: 找到最近的车道
        closest_lane = None
        min_lateral = float('inf')

        for start_node in network.graph:
            for end_node in network.graph[start_node]:
                for idx, lane in enumerate(network.graph[start_node][end_node]):
                    longitudinal, lateral = lane.local_coordinates(position)
                    if abs(lateral) < abs(min_lateral):
                        min_lateral = lateral
                        closest_lane = lane

        if closest_lane is None:
            raise ValueError(f"在位置 ({x}, {y}) 未找到任何车道")

        # Step 2: 计算切向量
        longitudinal, lateral = closest_lane.local_coordinates(position)
        ds = 0.1
        p1 = closest_lane.position(longitudinal, lateral)
        p2 = closest_lane.position(longitudinal + ds, lateral)

        tangent_vector = np.array(p2) - np.array(p1)
        tangent_norm = np.linalg.norm(tangent_vector)

        if tangent_norm < 1e-6:
            tangent_unit = np.array([1.0, 0.0])
        else:
            tangent_unit = tangent_vector / tangent_norm

        # Step 3: 计算法向量（指向左侧）
        normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])

        # Step 4: 合成笛卡尔速度
        velocity_cartesian = v_lon * tangent_unit + v_lat * normal_unit
        vx, vy = velocity_cartesian[0], velocity_cartesian[1]

        return vx, vy



