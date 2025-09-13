from typing import Optional

from urm.config import Config
from urm.reward.reward_meta import RewardMeta
from urm.reward.riskmap.risk_map import RiskMap
from urm.reward.riskmap.riskmap_manager import RiskMapManager
from urm.reward.state.ego_state import EgoState
from urm.reward.state.interface import EnvInterface
from urm.reward.state.state import State
from urm.reward.state.surrounding_state import SurroundingState
from urm.reward.trajectory.behavior import BehaviorFactory
from urm.reward.trajectory.trajectory_generator import TrajectoryGenerator
from urm.reward.trajectory.prediction import *


class RiskMapReward(RewardMeta):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.riskmap_manager = Optional[RiskMapManager]
        self.behavior_factory = BehaviorFactory(config.reward.behavior_configs)
        self.prediction_model = create_model_from_config(self.config)
        self.behaviors = self.behavior_factory.get_all_behaviors_by_config()

    def reward(self, ego_state: EgoState, surrounding_states: SurroundingState, env_condition: EnvInterface,
               baseline_reward):
        assert self.riskmap_manager is not None, "riskmap_manager should be initialized"

        self.riskmap_manager_create(ego_state=ego_state, surrounding_states=surrounding_states,
                                    env_condition=env_condition)
        riskmap_total: RiskMap = self.riskmap_manager.sum_all()
        return self.urm_risk(custom_risk=riskmap_total.get_risk_for_car(ego_state, self.riskmap_manager.world_to_local),
                             baseline=baseline_reward)

    def urm_risk(self, custom_risk, baseline):
        return self.config.reward.baseline_reward_w * baseline + (
                1 - self.config.reward.baseline_reward_w) * custom_risk

    def riskmap_manager_create(self, ego_state: EgoState, surrounding_states: SurroundingState,
                               env_condition: EnvInterface):
        global_state = State(env=env_condition)
        trajs = TrajectoryGenerator(ego_state, surrounding_states, env_condition=global_state,
                                    behaviors=self.behaviors,
                                    prediction_model=self.prediction_model, config=self.config).generate_right(
            self.config.reward.step_num,
            self.config.reward.duration)
        self.riskmap_manager = RiskMapManager(config=self.config.reward, trajtree=trajs)
