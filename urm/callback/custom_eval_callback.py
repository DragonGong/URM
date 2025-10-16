import logging
import os
import pprint

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
import numpy as np
from typing import Any, Dict, List
import torch as th

from urm import utils


class CustomEvalCallback(EvalCallback):
    def __init__(
            self,
            eval_env,
            best_model_save_path=None,
            best_model_name="best_model",  # 自定义文件名
            log_path=None,
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1,
            best_metric_fn=None,  # 自定义“best”判定函数
            config=None
    ):
        super().__init__(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )
        self.best_model_name = best_model_name
        self.best_metric_fn = best_metric_fn or self._default_best_metric_fn

        # 存储历史评估结果（用于 best 判定）
        self.eval_results = {}

        # 初始化 best 指标为 -inf
        self.best_metric_value = -np.inf
        self.config = config

    @staticmethod
    def _default_best_metric_fn(eval_results):
        """默认：用 mean reward 作为 best 指标"""
        return eval_results["mean_reward"]

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset episode buffers
            episode_crashes = []
            episode_speeds = []
            episode_jerks = []
            episode_successes = []
            episode_rewards = []

            sync_envs_normalization(self.training_env, self.eval_env)

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                ep_speeds = []
                ep_jerks = []
                ep_reward = 0.0
                crashed = False
                success = False
                last_acc = 0.0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, done, info = self.eval_env.step(action)
                    # done = terminated or truncated

                    ep_reward += reward

                    # Unwrap info for VecEnv
                    info = info[0] if isinstance(info, (list, tuple)) else info

                    speed = info.get("speed", 0.0)
                    ep_speeds.append(speed)

                    acc = info.get("acceleration", 0.0)
                    jerk = abs(acc - last_acc)
                    ep_jerks.append(jerk)
                    last_acc = acc

                    if info.get("crashed", False):
                        crashed = True
                    if info.get("is_success", False):
                        success = True

                # End of episode
                episode_crashes.append(crashed)
                episode_speeds.append(np.mean(ep_speeds) if ep_speeds else 0.0)
                episode_jerks.append(np.mean(ep_jerks) if ep_jerks else 0.0)
                episode_successes.append(success)
                episode_rewards.append(ep_reward)  # 一个episode累计的reward

            # Compute metrics
            avg_collision_rate = np.mean(episode_crashes)
            avg_speed = np.mean(episode_speeds)
            avg_jerk = np.mean(episode_jerks)
            success_rate = np.mean(episode_successes)
            mean_reward = np.mean(episode_rewards)

            # Save all results for best_metric_fn
            self.eval_results = {
                "mean_reward": float(mean_reward),
                "collision_rate": float(avg_collision_rate),
                "avg_speed": float(avg_speed),
                "avg_jerk": float(avg_jerk),
                "success_rate": float(success_rate),
            }

            # Log to TensorBoard
            if self.model.logger:
                self.model.logger.record("eval/collision_rate", avg_collision_rate)
                self.model.logger.record("eval/avg_speed", avg_speed)
                self.model.logger.record("eval/avg_jerk", avg_jerk)
                self.model.logger.record("eval/success_rate", success_rate)
                self.model.logger.record("eval/mean_reward", mean_reward)

            # Determine if this is the best model
            current_metric = self.best_metric_fn(self.eval_results)

            if current_metric > self.best_metric_value:
                self.best_metric_value = current_metric
                if self.best_model_save_path is not None:
                    save_path = os.path.join(
                        self.best_model_save_path,
                        f"{self.best_model_name}.zip"
                    )
                    self.model.save(save_path)
                    utils.write_config_to_file(self.config,
                                               os.path.join(self.best_model_save_path, f"{self.best_model_name}.txt"))
                    if self.verbose >= 1:
                        logging.info(f"New best model ({self.best_metric_fn.__name__}): "
                                     f"{current_metric:.4f} → saved to {save_path}")

        return continue_training
