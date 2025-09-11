import numpy as np

from urm.reward.state.car_state import CarState


class HighwayState:
    def __init__(self, vehicle, duration: int = 1, time_step: int = 1):
        """
        vehicle: highway-env 的 Vehicle 对象
        """
        self.position = vehicle.position  # (x, y)
        self.velocity = np.linalg.norm(vehicle.velocity)  # 标量速度
        self.acceleration = vehicle.acceleration  # 可能是标量或向量，看需求
        self.orientation = vehicle.heading  # 朝向
        self.time_step = time_step
        self.duration = duration

    @classmethod
    def from_carstate(cls, car_state: CarState, duration: int):
        raise NotImplementedError
