from abc import ABC, abstractmethod

from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from urm.reward.trajectory.highway_env_state import HighwayState as State


class Behavior(ABC):

    @abstractmethod
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass
        # =============================


# Lateral Behaviors
# =============================
class LateralBehavior(Behavior):
    """Base class for lateral behaviors."""

    @abstractmethod
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    @abstractmethod
    def target_offset(self, root_offset: float, state: State) -> float:
        pass


class LateralKeep(LateralBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset


class LateralLeft(LateralBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset - (0.3 if state.velocity < 5 else 0.5)


class LateralRight(LateralBehavior):
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset + (0.3 if state.velocity < 5 else 0.5)


# =============================
# Longitudinal Behaviors
# =============================
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
        pass

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
