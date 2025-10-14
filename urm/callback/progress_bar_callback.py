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
