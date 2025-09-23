import numpy as np
import gymnasium as gym
from urm.reward.state.car_state import CarState
from highway_env.vehicle.kinematics import Vehicle

from urm.reward.state.utils.position import Position
from urm.reward.state.utils.velocity import Velocity


class HighwayState:
    """
    用来兼容lqy behavior代码
    """

    def __init__(self, vehicle: Vehicle, duration: int = 1, time_step: int = 1):
        """
        vehicle: highway-env 的 Vehicle 对象
        """
        self.position = Position.from_nparray(vehicle.position)  # (x, y)
        vx, vy = vehicle.velocity
        self.velocity = Velocity(vx=vx, vy=vy)
        self.acceleration = vehicle.action.get("acceleration", None)
        self.orientation = vehicle.heading  # 朝向
        self.time_step = time_step
        self.duration = duration
        self.env = None

    @classmethod
    def from_carstate(cls, car_state: CarState, duration: int, time_step=3):
        instance = cls.__new__(cls)
        instance.position = Position.from_nparray(np.array([car_state.x, car_state.y]))
        instance.velocity = car_state.velocity
        instance.acceleration = None
        instance.orientation = np.arctan2(car_state.vy, car_state.vx)
        instance.time_step = time_step
        instance.duration = duration
        instance.env = car_state.env_condition
        return instance
