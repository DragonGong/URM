import gymnasium as gym


# 改了一点，但是大部分gpt
class BaselineEnv(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
