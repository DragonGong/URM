import sys

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    """
    显示训练进度条的回调
    """

    def __init__(self, total_timesteps: int, name: str = None):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None
        if name is None:
            self.name = "Training"
        else:
            self.name = name

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc=self.name, unit="step")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


class MultiProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, position: int, name: str = None):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.position = position  # ← 关键：指定行号
        self.name = name or f"Seed {position}"
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc=self.name,
            unit="step",
            position=self.position,  # 固定行位置
            leave=True,  # 训练完保留最后一行
            file=sys.stderr,  # ← 输出到 stderr，不进日志文件
            dynamic_ncols=True,  # 自动适应终端宽度
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
