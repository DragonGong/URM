import logging

from stable_baselines3.common.callbacks import CallbackList
import os
import datetime
import pprint

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from urm.callback.progress_bar_callback import ProgressBarCallback
from urm.config import Config
from urm.env_wrapper.baseline_env import BaselineEnv
import highway_env

from urm.env_wrapper.env_factory import make_wrapped_env
from urm.callback.custom_eval_callback import CustomEvalCallback

# 支持的算法映射
ALGORITHM_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}


# renovation by Qwen

def make_env(config, render_mode=None):
    """根据 Config 对象创建环境"""
    env_config = config.env_config
    env = gym.make(env_config.env_id, render_mode=render_mode)

    idm_config = getattr(env_config, 'IDMVehicle', {})
    if idm_config:
        setattr(env_config, 'highway_env.vehicle.behavior.IDMVehicle', idm_config)

    env.unwrapped.configure(env_config.__dict__)  # 传入字典配置
    env.reset()
    env = make_wrapped_env(env, config)
    return env


def train_model(config: Config):
    """
    训练函数，接收 Config 对象，自动选择算法并训练

    :param config: Config 实例（包含 env_config, model_config, training 等）
    """
    env = DummyVecEnv([lambda: make_env(config, config.training.render_mode)])

    algo_name = config.model_config.algorithm  # 如 "DQN", "PPO"
    if algo_name not in ALGORITHM_MAP:
        raise ValueError(f"Unsupported algorithm: {algo_name}. Supported: {list(ALGORITHM_MAP.keys())}")

    model_class = ALGORITHM_MAP[algo_name]

    model_kwargs = {
        "policy": config.model_config.policy,
        "env": env,
        "verbose": config.model_config.verbose,
    }

    common_params = [
        "learning_rate",
        "gamma",
        "tensorboard_log",
    ]

    algo_specific_params = {
        "DQN": [
            "buffer_size",
            "learning_starts",
            "batch_size",
            "train_freq",
            "gradient_steps",
            "target_update_interval",
            "policy_kwargs",
        ],
        "PPO": [
            "n_steps",
            "batch_size",
            "n_epochs",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "policy_kwargs",
        ],
        "A2C": [
            "n_steps",
            "gae_lambda",
            "ent_coef",
            "policy_kwargs",
        ],
    }

    for param in common_params:
        if hasattr(config.model_config, param) and getattr(config.model_config, param) is not None:
            value = getattr(config.model_config, param)
            if param == "tensorboard_log" and value:
                value = os.path.join(value, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            model_kwargs[param] = value

    for param in algo_specific_params.get(algo_name, []):
        if hasattr(config.model_config, param) and getattr(config.model_config, param) is not None:
            model_kwargs[param] = getattr(config.model_config, param)

    algo_name = config.model_config.algorithm
    task_name = config.env_config.env_id.replace("/", "_")
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    best_model_name = f"{algo_name}_{task_name}_{timestamp}_best"

    eval_callback = CustomEvalCallback(
        env,
        best_model_save_path=os.path.join(config.training.save_dir, "best_model"),
        best_model_name=best_model_name,
        log_path=config.training.save_dir,
        eval_freq=config.training.eval_freq,  # e.g., 100 steps
        n_eval_episodes=config.training.n_eval_episodes,  # 每次评估跑 10 个 episode
        deterministic=True,
        render=False,
    )
    progress_bar_callback = ProgressBarCallback(total_timesteps=config.training.total_timesteps)
    callback_list = CallbackList([eval_callback, progress_bar_callback])
    logging.info(f"Creating model: {algo_name} with params:")
    pprint.pprint(model_kwargs)
    model = model_class(**model_kwargs)

    logging.info("Environment Config:")
    pprint.pprint(env.envs[0].unwrapped.config)

    logging.info(f"🚀 Starting training with {algo_name}...")
    model.learn(total_timesteps=config.training.total_timesteps, log_interval=1, callback=callback_list)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"{timestamp}_{algo_name.lower()}_baseline_urm_highway"
    save_path = os.path.join(config.training.save_dir, model_filename)
    os.makedirs(config.training.save_dir, exist_ok=True)
    model.save(save_path)

    logging.info(f"✅ Model saved to: {save_path}")
    return model, save_path
