import datetime
import os.path
import time
import numpy as np
import highway_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from urm.env_wrapper.urm_env import URMHighwayEnv

if __name__ == "__main__":
    def make_env():
        env = gym.make("highway-v0")
        env.unwrapped.configure({
            "lanes_count": 4,
            "vehicles_count": 20,
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "observation": {
                "type": "Kinematics"
            },
            "vehicles_density": 1.2,
            "vehicles_types": ["highway_env.vehicle.behavior.IDMVehicle"],
            "highway_env.vehicle.behavior.IDMVehicle": {
                "target_speed": 10,  # 所有 NPC 车辆的目标速度
                "maximum_speed": 15,
                "minimum_speed": 5
            }
        })
        env.reset()
        return URMHighwayEnv(env)


    env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, device="cpu")

    print("开始训练...")
    model.learn(total_timesteps=300000)
    model_filename = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_ppo_urm_highway"
    model.save(os.path.join("./agent", model_filename))
