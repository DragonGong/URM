import gymnasium as gym

from urm.reward.reward_meta import RewardMeta
from urm.reward.state.ego_state import EgoState
from urm.reward.state.surrounding_state import SurroundingState
from urm.env_wrapper.env import Env


class RiskMapEnv(Env):
    def __init__(self, env, reward: RewardMeta):
        super().__init__(env)
        self.reward = reward

    def step(self, action):
        obs, base_line_reward, terminated, truncated, info = self.env.step(action)

        env = self.env.unwrapped

        ego_state = EgoState.from_vehicle(env.vehicle, env=env)
        surrounding_state = SurroundingState.from_road_vehicles(road_vehicles=env.road.vehicles,
                                                                exclude_vehicle=env.vehicle, env=env)
        reward = self.reward.reward(ego_state, surrounding_state, self, base_line_reward)
        return obs, reward, terminated, truncated, info
