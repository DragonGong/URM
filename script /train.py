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
            "vehicles_density": 1.0
        })
        env.reset()
        return URMHighwayEnv(env)


    env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99)

    print("开始训练...")
    model.learn(total_timesteps=200000)
    model.save("ppo_urm_highway")

    # 测试
    test_env = make_env()
    obs = test_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        test_env.render()
        total_reward += reward
    print("测试总奖励:", total_reward)
