import datetime
import os
import pprint

import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from urm.env_wrapper.baseline_env import BaselineEnv
from urm.env_wrapper.urm_env import URMHighwayEnv
import highway_env
from stable_baselines3 import DQN


# 读取 YAML 配置
def load_config(config_path="config/config_baseline.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_env(config, render_mode=None):
    env_config = config['env_config']
    env = gym.make(env_config['env_id'], render_mode=render_mode)

    # 构建 IDMVehicle 配置
    idm_config = env_config.get('IDMVehicle', {})
    if idm_config:
        env_config['highway_env.vehicle.behavior.IDMVehicle'] = idm_config

    env.unwrapped.configure(env_config)
    env.reset()
    return BaselineEnv(env, config)


if __name__ == "__main__":
    # 加载配置
    config = load_config()

    # 创建环境
    env = DummyVecEnv([lambda: make_env(config)])

    # 从配置创建模型
    model_config = config['model_config']
    model = DQN(
        policy=model_config['policy'],
        env=env,
        policy_kwargs=model_config.get('policy_kwargs', {}),
        learning_rate=model_config['learning_rate'],
        buffer_size=model_config['buffer_size'],
        learning_starts=model_config['learning_starts'],
        batch_size=model_config['batch_size'],
        gamma=model_config['gamma'],
        train_freq=model_config['train_freq'],
        gradient_steps=model_config['gradient_steps'],
        target_update_interval=model_config['target_update_interval'],
        verbose=model_config['verbose'],
        tensorboard_log=os.path.join(model_config.get('tensorboard_log', None),
                                     datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    pprint.pprint(env.envs[0].unwrapped.config)
    print("开始训练...")
    training_config = config['training']
    model.learn(total_timesteps=training_config['total_timesteps'])

    # 保存模型
    model_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_baseline_dqn_urm_highway"
    save_path = os.path.join(training_config['save_dir'], model_filename)
    os.makedirs(training_config['save_dir'], exist_ok=True)  # 确保目录存在
    model.save(save_path)

    print(f"模型已保存至: {save_path}")
