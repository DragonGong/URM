from abc import ABC, abstractmethod

from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from urm.reward.trajectory.highway_env_state import HighwayState as State

from .behaviors import Behavior


class LongitudinalBehavior(Behavior):
    """Base class for longitudinal behaviors."""

    @abstractmethod
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    @abstractmethod
    def target_velocity(self, root_velocity: float, state: State) -> float:
        raise NotImplementedError


class LongitudinalCruise(LongitudinalBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        v = state.velocity
        vx, vy = v.vx, v.vy
        duration = state.duration

        dx = vx * duration
        dy = vy * duration

        new_x = initial_position.x + dx
        new_y = initial_position.y + dy

        target_car_state = CarState(
            x=new_x,
            y=new_y,
            vx=vx,
            vy=vy
        )
        return target_car_state

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return state.velocity


class SoftAcceleration(LongitudinalBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return min(state.velocity + 2.5, 50.0)


class HardAcceleration(LongitudinalBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return min(state.velocity + 5.0, 50.0)


class SoftDeceleration(LongitudinalBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return max(state.velocity - 3.0, 0.01)


class HardDeceleration(LongitudinalBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return max(state.velocity - 10.0, 0.01)
