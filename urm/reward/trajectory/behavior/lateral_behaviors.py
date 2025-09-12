from abc import ABC, abstractmethod

import numpy as np

from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from urm.reward.trajectory.highway_env_state import HighwayState as State

from .behaviors import Behavior


class LateralBehavior(Behavior):
    """Base class for lateral behaviors."""

    @abstractmethod
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    @abstractmethod
    def target_offset(self, root_offset: float, state: State) -> float:
        pass


class LateralKeep(LateralBehavior):
    # 暂时没用处
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset


class LateralLeft(LateralBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        v = state.velocity
        vx, vy = v.vx, v.vy
        duration = getattr(state, 'duration', 1.0)
        LATERAL_OFFSET = 0.5
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed < 1e-6:
            # 如果速度几乎为0，则无法定义方向，直接横向偏移
            dx, dy = 0.0, LATERAL_OFFSET
        else:
            forward_unit = np.array([vx, vy]) / speed
            lateral_unit = np.array([-vy, vx]) / speed

            # 纵向位移：沿速度方向走 duration 时间
            longitudinal_displacement = forward_unit * speed * duration
            # 横向位移：向左偏移 LATERAL_OFFSET
            lateral_displacement = lateral_unit * LATERAL_OFFSET

            total_displacement = longitudinal_displacement + lateral_displacement
            dx, dy = total_displacement[0], total_displacement[1]

        new_x = initial_position.x + dx
        new_y = initial_position.y + dy

        target_car_state = CarState(
            x=new_x,
            y=new_y,
            vx=vx,
            vy=vy
        )

        return target_car_state

    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset - (0.3 if state.velocity < 5 else 0.5)


class LateralRight(LateralBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        v = state.velocity
        vx, vy = v.vx, v.vy
        duration = getattr(state, 'duration', 1.0)
        LATERAL_OFFSET = 0.5
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed < 1e-6:
            # 如果速度几乎为0，则无法定义方向，直接横向偏移
            dx, dy = 0.0, LATERAL_OFFSET
        else:
            forward_unit = np.array([vx, vy]) / speed
            lateral_unit = np.array([vy, -vx]) / speed

            # 纵向位移：沿速度方向走 duration 时间
            longitudinal_displacement = forward_unit * speed * duration
            # 横向位移：向右边偏移 LATERAL_OFFSET
            lateral_displacement = lateral_unit * LATERAL_OFFSET

            total_displacement = longitudinal_displacement + lateral_displacement
            dx, dy = total_displacement[0], total_displacement[1]

        new_x = initial_position.x + dx
        new_y = initial_position.y + dy

        target_car_state = CarState(
            x=new_x,
            y=new_y,
            vx=vx,
            vy=vy
        )
        return target_car_state

    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset + (0.3 if state.velocity < 5 else 0.5)
