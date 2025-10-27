import os
import datetime
import time

import cv2

from urm.utils.constant import Mode
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from urm.config.config import Config
from urm.env_wrapper.env_factory import make_wrapped_env

ALGORITHM_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
}


def make_env(config, render_mode=None):
    env_config = config.env_config
    env = gym.make(env_config.env_id, render_mode=render_mode)
    if not config.env_config.default_config:
        idm_config = getattr(env_config, 'IDMVehicle', {})
        if idm_config:
            setattr(env_config, 'highway_env.vehicle.behavior.IDMVehicle', idm_config)
        env.unwrapped.configure(env_config.__dict__)
    env.reset()
    env = make_wrapped_env(env, config)
    logging.info(env.unwrapped.config)
    return env


def test_model(config):
    config.run_mode = Mode.TEST
    test_config = config.test_config

    test_env = DummyVecEnv([
        lambda: make_env(config, render_mode=test_config.render_mode)
    ])

    model_path = test_config.model_path

    # 推断算法
    algo_name = None
    if hasattr(config, 'model_config') and hasattr(config.model_config, 'algorithm'):
        algo_name = config.model_config.algorithm
    else:
        if "dqn" in model_path.lower():
            algo_name = "DQN"
        elif "ppo" in model_path.lower():
            algo_name = "PPO"

    if not algo_name or algo_name not in ALGORITHM_MAP:
        raise ValueError(f"无法识别算法，请检查配置或模型文件名。支持: {list(ALGORITHM_MAP.keys())}")

    model_class = ALGORITHM_MAP[algo_name]
    logging.info(f"📂 加载模型: {model_path} (算法: {algo_name})")
    model = model_class.load(model_path)

    logging.info("▶️ 开始测试...")
    total_reward = 0
    total_speed = 0
    total_risk = 0
    total_crashed = 0
    total_success = 0
    for episode in range(test_config.test_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0
        episode_risk = 0
        episode_crashed = False
        episode_speed = []
        episode_success = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            if test_config.render_mode is not None:
                if test_config.render_mode == "human":
                    test_env.render()
                else:
                    frame = test_env.render()
                    cv2.imshow("riskmap frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            episode_reward += reward[0]
            episode_risk += info[0]["risk"]
            if done and info[0]["crashed"]:
                episode_crashed = True
            if done and info[0]["is_success"]:
                episode_success = True
            episode_speed.append(info[0]["speed"])
            logging.info(f"action is {info[0]['action']}")
            step += 1
        logging.info(f"✅ Episode {episode + 1} | 步数: {step} | 总奖励: {episode_reward:.2f} | is_crashed: {episode_crashed} | is_success: {episode_success}")
        total_reward += episode_reward
        total_risk += episode_risk
        total_speed += sum(episode_speed)/len(episode_speed)
        if episode_success:
            total_success += 1
        if episode_crashed:
            total_crashed += 1

    avg_reward = total_reward / test_config.test_episodes
    avg_crashed = total_crashed / test_config.test_episodes
    avg_risk = total_risk / test_config.test_episodes
    avg_speed = total_speed / test_config.test_episodes
    avg_success = total_success / test_config.test_episodes
    logging.info(f"📈 平均奖励: {avg_reward:.2f}")
    logging.info(f"📈 碰撞率: {avg_crashed:.2f}")
    logging.info(f"📈 平均risk: {avg_risk:.2f}")
    logging.info(f"📈 平均速度: {avg_speed:.2f}")
    logging.info(f"📈 成功率: {avg_success:.2f}")
    test_env.close()
    return avg_reward
