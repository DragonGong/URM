from typing import Union, Tuple, Any

import gymnasium as gym
import numpy as np

from urm.config import Config
from urm.reward.state.utils.position import Position
from urm.reward.state.interface.env_interface import EnvInterface
import highway_env


class Env(gym.Wrapper):
    def __init__(self, env, config :Config, **kwargs):
        super().__init__(env)
        self.config = config

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
            pos: Union[Tuple[float, float], Position, np.ndarray, dict],
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


if __name__ == "__main__":
    env = Env(gym.make("highway-fast-v0"))
    print(isinstance(env, EnvInterface))  # → True ✅
