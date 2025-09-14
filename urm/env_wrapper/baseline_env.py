from typing import Union

import gymnasium as gym
from urm.env_wrapper.env import Env
from urm.config import Config


# 改了一点，但是大部分gpt
class BaselineEnv(Env):
    def __init__(self, env, config: Union[Config, dict], **kwargs):
        super().__init__(env, config, **kwargs)
        if isinstance(config, Config):
            self.config = config.to_dict()
        self.config = config

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
