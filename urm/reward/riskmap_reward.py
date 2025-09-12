from urm.config import Config
from urm.reward.reward_meta import RewardMeta
from urm.reward.riskmap.risk_map import RiskMap
from urm.reward.riskmap.map_parameter import MapParameter
from urm.reward.trajectory.behavior import BehaviorFactory
from urm.reward.trajectory.trajectory_generator import TrajectoryGenerator
from urm.reward.trajectory.prediction import *


class RiskMapReward(RewardMeta):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.riskmap_last = None
        self.behavior_factory = BehaviorFactory(config.reward.behavior_configs)
        pass

    def reward(self, ego_state, surrounding_states, env_condition, baseline_reward):
        ego_position = ego_state.get_position()
        assert self.riskmap_last is not None, "riskmap should be initialized"
        map_risk = self.riskmap_last.get_risk_via_position(ego_position)
        urm_risk = self._reward_calculate(map_risk, baseline_reward, self.config.reward)
        prediction_model = create_model_from_config(self.config)
        behaviors = self.behavior_factory.get_all_behaviors()
        trajs = TrajectoryGenerator(ego_state, surrounding_states, ego_state, behaviors=behaviors,
                                    prediction_model=prediction_model, config=self.config).generate_right(
            self.config.reward.step_num,
            self.config.reward.duration)

        map_param = MapParameter(self.config.get('size', {}), env_condition)
        self.riskmap_last = RiskMap(trajs, map_param)
        return urm_risk

    def _reward_calculate(self, map_risk, baseline_reward=None, reward_config=None):
        return 0

    def riskmap_initial(self):
        self.riskmap_last = ...
