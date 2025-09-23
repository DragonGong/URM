import math
from abc import ABC, abstractmethod

from urm.config import Config
from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from urm.reward.trajectory.highway_env_state import HighwayState as State
from .behavior_factory import BehaviorFactory

from .behaviors import Behavior
from .constant import BehaviorName


class LongitudinalBehavior(Behavior):
    """Base class for longitudinal behaviors."""

    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        super().__init__(config, **kwargs)

    @abstractmethod
    def target_state(self, initial_position: Position, state: State) -> CarState:
        pass

    @abstractmethod
    def target_velocity(self, root_velocity: float, state: State) -> float:
        raise NotImplementedError


@BehaviorFactory.register(BehaviorName.CRUISE)
class LongitudinalCruise(LongitudinalBehavior):
    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        super().__init__(config, **kwargs)

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
        return root_velocity


@BehaviorFactory.register(BehaviorName.SOFT_ACCEL)
class SoftAcceleration(LongitudinalBehavior):
    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        super().__init__(config, **kwargs)

    def target_state(self, initial_position: Position, state: State) -> CarState:
        v = state.velocity
        vx, vy = v.vx, v.vy
        duration = state.duration

        # 初始速度大小
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # 柔和加速度大小 (m/s^2)
        accel = 1.0

        if speed > 1e-6:  # 有速度，按方向加速
            ax = accel * (vx / speed)
            ay = accel * (vy / speed)
        else:  # 静止，默认往 x 正方向加速
            ax, ay = accel, 0.0

        # 匀加速公式更新速度
        new_vx = vx + ax * duration
        new_vy = vy + ay * duration

        # 匀加速公式更新位置
        dx = vx * duration + 0.5 * ax * duration ** 2
        dy = vy * duration + 0.5 * ay * duration ** 2

        new_x = initial_position.x + dx
        new_y = initial_position.y + dy

        target_car_state = CarState(
            x=new_x,
            y=new_y,
            vx=new_vx,
            vy=new_vy
        )
        return target_car_state

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return min(root_velocity + 2.5, 50.0)


@BehaviorFactory.register(BehaviorName.HARD_ACCEL)
class HardAcceleration(LongitudinalBehavior):
    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        super().__init__(config, **kwargs)

    def target_state(self, initial_position: Position, state: State) -> CarState:
        v = state.velocity
        vx, vy = v.vx, v.vy
        duration = state.duration

        # 初始速度大小
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # 硬加速，加速度更大
        accel = 3.5  # m/s^2，可以调节

        if speed > 1e-6:  # 有初始速度，保持方向
            ax = accel * (vx / speed)
            ay = accel * (vy / speed)
        else:  # 如果静止，默认沿 x 方向加速
            ax, ay = accel, 0.0

        # 匀加速更新速度
        new_vx = vx + ax * duration
        new_vy = vy + ay * duration

        # 匀加速更新位置
        dx = vx * duration + 0.5 * ax * duration ** 2
        dy = vy * duration + 0.5 * ay * duration ** 2

        new_x = initial_position.x + dx
        new_y = initial_position.y + dy

        target_car_state = CarState(
            x=new_x,
            y=new_y,
            vx=new_vx,
            vy=new_vy
        )
        return target_car_state

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return min(root_velocity + 5.0, 50.0)


@BehaviorFactory.register(BehaviorName.SOFT_DECEL)
class SoftDeceleration(LongitudinalBehavior):
    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        super().__init__(config, **kwargs)

    def target_state(self, initial_position: Position, state: State) -> CarState:
        v = state.velocity
        vx, vy = v.vx, v.vy
        duration = state.duration

        # 初始速度大小
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # 柔和减速
        decel = 2.0  # m/s^2

        if speed > 1e-6:
            ax = -decel * (vx / speed)
            ay = -decel * (vy / speed)
        else:  # 已经基本停下
            ax, ay = 0.0, 0.0

        # 匀减速更新速度
        new_vx = max(vx + ax * duration, 0.0)
        new_vy = max(vy + ay * duration, 0.0)

        # 匀减速更新位置
        dx = vx * duration + 0.5 * ax * duration ** 2
        dy = vy * duration + 0.5 * ay * duration ** 2

        new_x = initial_position.x + dx
        new_y = initial_position.y + dy

        return CarState(
            x=new_x,
            y=new_y,
            vx=new_vx,
            vy=new_vy
        )

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return max(root_velocity - 3.0, 0.01)


@BehaviorFactory.register(BehaviorName.HARD_DECEL)
class HardDeceleration(LongitudinalBehavior):
    def __init__(self, config: Config.RewardConfig.BehaviorConfigs, **kwargs):
        super().__init__(config, **kwargs)

    def target_state(self, initial_position: Position, state: State) -> CarState:
        v = state.velocity
        vx, vy = v.vx, v.vy
        duration = state.duration

        # 初始速度大小
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # 急刹车
        decel = 6.0  # m/s^2

        if speed > 1e-6:
            ax = -decel * (vx / speed)
            ay = -decel * (vy / speed)
        else:
            ax, ay = 0.0, 0.0

        # 匀减速更新速度
        new_vx = max(vx + ax * duration, 0.0)
        new_vy = max(vy + ay * duration, 0.0)

        # 匀减速更新位置
        dx = vx * duration + 0.5 * ax * duration ** 2
        dy = vy * duration + 0.5 * ay * duration ** 2

        new_x = initial_position.x + dx
        new_y = initial_position.y + dy

        return CarState(
            x=new_x,
            y=new_y,
            vx=new_vx,
            vy=new_vy
        )

    def target_velocity(self, root_velocity: float, state: State) -> float:
        return max(root_velocity - 10.0, 0.01)
