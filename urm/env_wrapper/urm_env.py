from typing import Union

import gymnasium as gym

from urm.config import Config
from urm.reward.urm_reward import URM_reward
from urm.env_wrapper.env import Env


# 改了一点，但是大部分gpt
class URMHighwayEnv(Env):
    def __init__(self, env, config: Union[Config, dict], **kwargs):
        super().__init__(env, **kwargs)
        if isinstance(config, Config):
            self.config = config.to_dict()
        self.config = config

    def step(self, action):
        obs, baseline_reward, terminated, truncated, info = self.env.step(action)

        env = self.env.unwrapped
        ego = [env.vehicle.position[0], env.vehicle.position[1],
               env.vehicle.velocity[0], env.vehicle.velocity[1]]

        surrounding = []
        for v in env.road.vehicles:
            if v is not env.vehicle:
                surrounding.append([v.position[0], v.position[1],
                                    v.velocity[0], v.velocity[1]])

        reward = URM_reward(ego, surrounding, self.config['reward'], baseline_reward)
        return obs, reward, terminated, truncated, info
