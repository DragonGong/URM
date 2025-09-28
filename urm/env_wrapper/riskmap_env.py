import gymnasium as gym

from urm.config import Config
from urm.reward.reward_meta import RewardMeta
from urm.reward.state.ego_state import EgoState
from urm.reward.state.surrounding_state import SurroundingState
from urm.env_wrapper.env import Env


class RiskMapEnv(Env):
    def __init__(self, env, config: Config, reward: RewardMeta, **kwargs):
        super().__init__(env, config, **kwargs)
        self.reward = reward
        self.last_acceleration = 0.0

    def step(self, action):
        obs, base_line_reward, terminated, truncated, info = self.env.step(action)
        if self.config.training.render_mode:
            self.env.render()
        env = self.env.unwrapped

        ego_state = EgoState.from_vehicle(env.vehicle, env=self)
        surrounding_state = SurroundingState.from_road_vehicles(road_vehicles=env.road.vehicles,
                                                                exclude_vehicle=env.vehicle, env=self)
        if self.config.reward.baseline_reward_w == 1 and self.config.reward.custom_reward_w == 0:
            reward = base_line_reward
        else:
            reward = self.reward.reward(ego_state, surrounding_state, self, base_line_reward,action)

        current_speed = env.vehicle.speed
        current_acceleration = env.vehicle.action["acceleration"] if hasattr(env.vehicle, 'action') else 0.0
        # jerk = |a_t - a_{t-1}|
        jerk = abs(current_acceleration - self.last_acceleration)
        self.last_acceleration = current_acceleration
        is_success = current_speed > 0 and env.vehicle.position[0] > 100  # 示例：x > 100m
        info.update({
            "speed": current_speed,
            "acceleration": current_acceleration,
            "jerk": jerk,
            "crashed": env.vehicle.crashed,
            "is_success": is_success,
            "on_road": env.vehicle.on_road,
        })
        return obs, reward, terminated, truncated, info

