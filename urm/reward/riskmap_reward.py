class RiskMapReward:
    def __init__(self, config):
        pass

    def reward(self, ego_state, surrounding_states, riskmap_last):
        ego_position = ego_state.get_position()
        map_risk = riskmap_last.get_risk_via_position(ego_position)
        urm_risk = self._reward_calculate(map_risk)
        return urm_risk

    def _reward_calculate(self, map_risk):
        return 0
