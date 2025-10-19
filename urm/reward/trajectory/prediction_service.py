from typing import Optional

from urm.config import Config
from urm.reward.state.surrounding_state import SurroundingState
from urm.reward.trajectory.prediction import create_model_from_config
from collections import deque


class PredictionService:
    def __init__(self, config: Config, interval: float = 0.1, total_time: float = 3, step_frequency=1, **kwargs):
        self.config = config
        self.prediction_model = create_model_from_config(config)
        self.interval = interval
        self.timestamp_num = int(total_time / self.interval)
        self.result = []
        self.surrounding_cars = None
        self.step_frequency = 1

    def get_surround_cars(self, radius: float, x: float, y: float, surround_states: SurroundingState):
        self.surrounding_cars = surround_states.get_cars_in_radius(x, y, radius)

    def set_step_frequency(self, step_frequency: int):
        self.step_frequency = step_frequency

    def update_result(self, radius: float, x: float, y: float, surround_states: SurroundingState):
        self.get_surround_cars(radius, x, y, surround_states)
        if self.result is None:
            self.result = []
        current_len = len(self.result)
        target_len = self.timestamp_num
        if current_len >= target_len:
            time_loss = int(1 / self.step_frequency)
            self.result = self.result[int(time_loss / self.interval):]
            current_len = target_len - int(time_loss/self.interval)
        num_to_predict = target_len - current_len
        new_snapshots = []
        for step in range(1, num_to_predict + 1):
            time_horizon = (current_len + step) * self.interval
            snapshot = []
            for car in self.surrounding_cars:
                region = self.prediction_model.predict_region(
                    car,
                    time_horizon,
                    width=car.vehicle_size.width,
                    length=car.vehicle_size.length
                )
                snapshot.append(region)
            new_snapshots.append(snapshot)
        self.result.extend(new_snapshots)
        assert len(self.result) == self.timestamp_num


    def FIFO_update(self, result):
        if self.result is None:
            self.result = []
        if len(self.result) >= self.timestamp_num:
            self.result.pop(0)
        self.result.append(result)
