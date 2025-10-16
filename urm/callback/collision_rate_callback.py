import logging

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from typing import Any


class CollisionRateCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.collision_buffer = deque(maxlen=window_size)
        self.speed_buffer = deque(maxlen=window_size)
        self.episode_count = 0
        self.speed_list = []

    def _on_step(self) -> bool:
        # SB3 在每个 step 结束时调用 _on_step
        # 我们通过 self.locals 获取当前 step 的 info
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                self.speed_list.append(info.get("speed", 0))
                if done:  # episode 结束
                    self.episode_count += 1
                    crashed = info.get("crashed", False)
                    self.collision_buffer.append(crashed)

                    if len(self.speed_list) == 0:
                        logging.error("speed list length is 0")
                        self.speed_buffer.append(0)
                    else:
                        self.speed_buffer.append(sum(self.speed_list) / len(self.speed_list))
                    self.speed_list = []
                    # 计算当前滑动窗口碰撞率
                    collision_rate = sum(self.collision_buffer) / len(self.collision_buffer)
                    mean_speed = sum(self.speed_buffer) / len(self.speed_buffer)

                    # 记录到 logger（自动关联到 model）
                    self.logger.record("train/collision_rate", collision_rate)
                    self.logger.record("train/mean_speed", mean_speed)
                    # 可选：也记录 episode 总 reward（SB3 默认已记录）
        return True
