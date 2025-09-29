

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    """
    显示训练进度条的回调
    """
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()