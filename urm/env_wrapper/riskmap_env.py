import logging
import time

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
        original_start = time.time()
        start = time.time()
        obs, base_line_reward, terminated, truncated, info = self.env.step(action)
        logging.debug("\n\n\n")
        logging.debug(f"the baseline step time consuming is {time.time() - start}s")
        start = time.time()
        if self.config.training.render_mode:
            self.env.render()
        env = self.env.unwrapped

        ego_state = EgoState.from_vehicle(env.vehicle, env=self)
        surrounding_state = SurroundingState.from_road_vehicles(road_vehicles=env.road.vehicles,
                                                                exclude_vehicle=env.vehicle, env=self)

        logging.debug(f"the state transfer time consuming is {time.time() - start}s")
        start = time.time()

        if (self.config.reward.version == 0 and (
                self.config.reward.baseline_reward_w == 1 and self.config.reward.custom_reward_w == 0)) \
                or (self.config.reward.version == 1 and (
                self.config.reward.baseline_reward_w == 1 and self.config.reward.risk_reward_w == 0)):
            reward = base_line_reward
        else:
            reward = self.reward.reward(ego_state, surrounding_state, self, base_line_reward, action)
            logging.debug(f"the reward calculation time is {time.time() - start}s")

        current_speed = env.vehicle.speed
        current_acceleration = env.vehicle.action["acceleration"] if hasattr(env.vehicle, 'action') else 0.0
        # jerk = |a_t - a_{t-1}|
        jerk = abs(current_acceleration - self.last_acceleration)
        self.last_acceleration = current_acceleration
        if "is_success" not in info:
            logging.debug("is_success is not in info")
            is_success = (
                    not info.get("crashed", False) and
                    terminated  # 只有正常结束才算成功（非 crash 导致的 terminated）
            )
        else:
            is_success = info.get("is_success", False)
        info.update({
            "speed": current_speed,
            "acceleration": current_acceleration,
            "jerk": jerk,
            "crashed": env.vehicle.crashed,
            "is_success": is_success,
            "on_road": env.vehicle.on_road,
        })
        logging.debug(f"the step last for {time.time() - original_start}s")
        return obs, reward, terminated, truncated, info
