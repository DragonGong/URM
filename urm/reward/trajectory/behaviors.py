from urm.reward.trajectory.highway_env_state import HighwayState as State


# =============================
# Lateral Behaviors
# =============================
class LateralBehavior:
    """Base class for lateral behaviors."""

    def target_offset(self, root_offset: float, state: State) -> float:
        raise NotImplementedError


class LateralKeep(LateralBehavior):
    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset


class LateralLeft(LateralBehavior):
    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset - (0.3 if state.velocity < 5 else 0.5)


class LateralRight(LateralBehavior):
    def target_offset(self, root_offset: float, state: State) -> float:
        return root_offset + (0.3 if state.velocity < 5 else 0.5)


# =============================
# Longitudinal Behaviors
# =============================
class LongitudinalBehavior:
    """Base class for longitudinal behaviors."""

    def target_velocity(self, root_velocity: float, state: State) -> float:
        raise NotImplementedError


class LongitudinalCruise(LongitudinalBehavior):
    def target_velocity(self, root_velocity: float, state: State) -> float:
        return state.velocity


class SoftAcceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity: float, state: State) -> float:
        return min(state.velocity + 2.5, 50.0)


class HardAcceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity: float, state: State) -> float:
        return min(state.velocity + 5.0, 50.0)


class SoftDeceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity: float, state: State) -> float:
        return max(state.velocity - 3.0, 0.01)


class HardDeceleration(LongitudinalBehavior):
    def target_velocity(self, root_velocity: float, state: State) -> float:
        return max(state.velocity - 10.0, 0.01)
