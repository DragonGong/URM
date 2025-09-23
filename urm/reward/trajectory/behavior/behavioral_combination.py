from abc import ABC
from typing import Optional

from urm.config import Config
from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from .behaviors import Behavior
from .longitudinal_behavior import LongitudinalBehavior
from .lateral_behaviors import LateralBehavior
from urm.reward.trajectory.highway_env_state import HighwayState as State
from urm.reward.utils import calculate_displacement


class BehavioralCombination(Behavior):

    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        super().__init__(config, **kwargs)
        self.longitudinal: Optional[LongitudinalBehavior] = None
        self.lateral: Optional[LateralBehavior] = None

    def target_state(self, initial_position: Position, state: State) -> CarState:
        assert self.lateral is not None and self.longitudinal is not None, ("self.lateral is None and "
                                                                            "self.longitudinal is None.")
        if not initial_position.calculated_frenet:
            initial_position.calculate_frenet(env=state.env)
        if not state.velocity.calculated_frenet:
            state.velocity.set_frenet(state.env, initial_position.x, initial_position.y)
        p_lat_after_offset = self.lateral.target_offset(initial_position.lateral, state)
        v_lon_ultimately = self.longitudinal.target_velocity(state.velocity.v_lot, state)

        initial_v_lon = state.velocity.v_lot
        displacement = calculate_displacement(initial_velocity=initial_v_lon, final_velocity=v_lon_ultimately,
                                              time=state.duration)

        p_lon = initial_position.longitudinal + displacement

        x, y = state.env.lane_local_to_world(lane_id=initial_position.lane_id, longitudinal=p_lon,
                                             lateral=p_lat_after_offset)

        # 预测的轨迹点的速度，都是沿着车道方向的，垂直于车道方向的速度全部归零
        v_lat_ultimately = 0.0

        vx, vy = state.env.frenet_velocity_to_cartesian(x=x, y=y, v_lon=v_lon_ultimately, v_lat=v_lat_ultimately)

        car_state = CarState(x=x, y=y, vx=vx, vy=vy, env=state.env)
        car_state.set_frenet_velocity()
        return car_state
