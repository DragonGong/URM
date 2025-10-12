import os
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class BestRewardCallback(BaseCallback):
    def __init__(self, save_path: str, window_size: int = 100, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.window_size = window_size
        self.reward_buffer = deque(maxlen=window_size)
        self.best_mean_reward = -float('inf')  # 初始化为负无穷

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                # 从 Monitor 注入的 info 中获取 episode 总 reward
                if "episode" in info:
                    episode_reward = info["episode"]["r"]
                    self.reward_buffer.append(episode_reward)

                    # 计算滑动窗口平均 reward
                    current_mean_reward = sum(self.reward_buffer) / len(self.reward_buffer)

                    # 如果比历史 best 更好，保存模型
                    if current_mean_reward > self.best_mean_reward:
                        self.best_mean_reward = current_mean_reward
                        # 保存模型（注意：路径需包含文件名，不含 .zip）
                        self.model.save(self.save_path)
                        if self.verbose > 0:
                            print(
                                f"🎉 New best mean reward: {current_mean_reward:.2f} - Model saved to {self.save_path}")

                    self.logger.record("train/best_mean_reward", self.best_mean_reward)
                    self.logger.record("train/current_mean_reward", current_mean_reward)

        return True
