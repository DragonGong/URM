import datetime
import os
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from urm.env_wrapper.urm_env import URMHighwayEnv
import highway_env


def load_config(config_path="config/train_config.yaml"):
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
    return URMHighwayEnv(env, config)


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
    model = PPO.load(model_path)

    print(f"加载模型: {model_path}")
    print("开始测试...")

    obs = test_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()  # 可视化
        total_reward += reward[0]  # VecEnv 返回的是数组

    print("✅ 测试完成！")
    print("📊 测试总奖励:", total_reward)

    test_env.close()
