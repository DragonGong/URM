import numpy as np
import highway_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from urm.env_wrapper.urm_env import URMHighwayEnv


def make_env():
    env = gym.make("highway-v0", render_mode='human')
    env.unwrapped.configure({
        "lanes_count": 4,
        "vehicles_count": 200,
        "duration": 40,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "observation": {
            "type": "Kinematics"
        },
        "vehicles_density": 1.5,
        "vehicles_types": ["highway_env.vehicle.behavior.IDMVehicle"],
        "highway_env.vehicle.behavior.IDMVehicle": {
            "target_speed": 10,  # 所有 NPC 车辆的目标速度
            "maximum_speed": 15,
            "minimum_speed": 5
        }

    })
    env.reset()
    return URMHighwayEnv(env)


# 加载模型
model = PPO.load("./agent/2025-09-03 13:53:44_ppo_urm_highway")
if __name__ == "__main__":
    # 测试
    test_env = DummyVecEnv([make_env])
    obs = test_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        total_reward += reward
    print("测试总奖励:", total_reward)
