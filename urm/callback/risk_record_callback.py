import os

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import logging


class RiskRecordCallback(BaseCallback):
    def __init__(self, window_size: int = 100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.risk_buffer = deque(maxlen=window_size)
        self.total_risk_sum = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                episode_risk = info.get("risk", 0.0)
                self.risk_buffer.append(episode_risk)
                self.total_risk_sum += episode_risk
                self.episode_count += 1

                current_mean_risk = sum(self.risk_buffer) / len(self.risk_buffer)

                self.logger.record("train/episode_risk", episode_risk)
                self.logger.record("train/mean_risk", current_mean_risk)
                # self.logger.record("risk/total_risk_sum", self.total_risk_sum)
                # self.logger.record("risk/episode_count", self.episode_count)

                if self.verbose > 0:
                    logging.info(
                        f"Episode {self.episode_count}: risk={episode_risk:.2f}, "
                        f"mean_risk (last {len(self.risk_buffer)})={current_mean_risk:.2f}"
                    )

        return True
