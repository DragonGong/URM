import numpy as np


class HighwayState:
    def __init__(self, vehicle, time_step: int = 0):
        """
        vehicle: highway-env 的 Vehicle 对象
        """
        self.position = vehicle.position  # (x, y)
        self.velocity = np.linalg.norm(vehicle.velocity)  # 标量速度
        self.acceleration = vehicle.acceleration  # 可能是标量或向量，看需求
        self.orientation = vehicle.heading  # 朝向
        self.time_step = time_step
