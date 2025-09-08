import gymnasium as gym
from urm.reward.urm_reward import URM_reward


# 改了一点，但是大部分gpt
class URMHighwayEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        env = self.env.unwrapped
        ego = [env.vehicle.position[0], env.vehicle.position[1],
               env.vehicle.velocity[0], env.vehicle.velocity[1]]

        surrounding = []
        for v in env.road.vehicles:
            if v is not env.vehicle:
                surrounding.append([v.position[0], v.position[1],
                                    v.velocity[0], v.velocity[1]])

        reward = URM_reward(ego, surrounding)
        return obs, reward, terminated, truncated, info


