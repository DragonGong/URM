import datetime
import logging
import os
import yaml
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from urm.env_wrapper.baseline_env import BaselineEnv
from urm.env_wrapper.urm_env import URMHighwayEnv
import highway_env


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
    test_config = config['test_config']

    # 创建测试环境（使用测试指定的 render_mode）
    test_env = DummyVecEnv([
        lambda: make_env(config, render_mode=test_config['render_mode'])
    ])

    # 加载模型
    model_path = test_config['model_path']
    model = DQN.load(model_path)

    logging.INFO(f"加载模型: {model_path}")
    print("开始测试...")

    total_reward = 0
    for episode in range(test_config['test_episodes']):
        obs = test_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            test_env.render()
            episode_reward += reward[0]
        print(f"Episode {episode + 1} 总奖励: {episode_reward}")
        total_reward += episode_reward

    avg_reward = total_reward / test_config['test_episodes']
    print(f"📈 平均奖励: {avg_reward}")

    test_env.close()
