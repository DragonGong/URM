from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
import numpy as np
from typing import Any, Dict, List
import torch as th


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_crashes = []
        self.episode_speeds = []
        self.episode_jerks = []
        self.episode_successes = []

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset lists
            self.episode_crashes = []
            self.episode_speeds = []
            self.episode_jerks = []
            self.episode_successes = []

            # Evaluate
            sync_envs_normalization(self.training_env, self.eval_env)
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                ep_speeds = []
                ep_jerks = []
                crashed = False
                success = False
                last_acc = 0.0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, done, info = self.eval_env.step(action)

                    # Collect info
                    info = info[0] if isinstance(info, list) else info  # unwrap if VecEnv

                    speed = info.get("speed", 0.0)
                    ep_speeds.append(speed)

                    # Compute jerk if acceleration is available
                    acc = info.get("acceleration", 0.0)
                    jerk = abs(acc - last_acc)
                    ep_jerks.append(jerk)
                    last_acc = acc

                    if info.get("crashed", False):
                        crashed = True
                    if info.get("is_success", False):
                        success = True

                # End of episode
                self.episode_crashes.append(crashed)
                self.episode_speeds.append(np.mean(ep_speeds))
                self.episode_jerks.append(np.mean(ep_jerks))
                self.episode_successes.append(success)

            # Log to TensorBoard
            if self.model.logger:
                avg_collision_rate = np.mean(self.episode_crashes)
                avg_speed = np.mean(self.episode_speeds)
                avg_jerk = np.mean(self.episode_jerks)
                success_rate = np.mean(self.episode_successes)

                self.model.logger.record("eval/collision_rate", avg_collision_rate)
                self.model.logger.record("eval/avg_speed", avg_speed)
                self.model.logger.record("eval/avg_jerk", avg_jerk)
                self.model.logger.record("eval/success_rate", success_rate)

        return continue_training
