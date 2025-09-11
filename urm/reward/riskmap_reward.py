from urm.reward.reward_meta import RewardMeta
from urm.reward.riskmap.risk_map import RiskMap
from urm.reward.riskmap.map_parameter import MapParameter
from urm.reward.trajectory.trajectory_generator import TrajectoryGenerator


class RiskMapReward(RewardMeta):
    def __init__(self, config):
        self.config = config
        self.riskmap_last = None
        pass

    def reward(self, ego_state, surrounding_states, env_condition, baseline_reward):
        ego_position = ego_state.get_position()
        assert self.riskmap_last is not None, "riskmap should be initialized"
        map_risk = self.riskmap_last.get_risk_via_position(ego_position)
        urm_risk = self._reward_calculate(map_risk, baseline_reward, self.config.get('reward', None))
        trajs = TrajectoryGenerator(ego_state, surrounding_states,ego_state, behaviors=behaviors,
                                    prediction_model=prediction_model).generate_right(
            self.config['step_nums'],
            self.config['duration'])
        map_param = MapParameter(self.config.get('size', {}), env_condition)
        self.riskmap_last = RiskMap(trajs, map_param)

        return urm_risk

    def _reward_calculate(self, map_risk, baseline_reward=None, reward_config=None):
        return 0

    def riskmap_initial(self):
        self.riskmap_last = ...
