import copy

from urm.reward.state.car_state import CarState


class IDMState(CarState):
    def __init__(self, env, original_vehicle, x=0.0, y=0.0, vx=0.0, vy=0.0, **kwargs):
        super().__init__(env, x, y, vx, vy, **kwargs)
        self.original_vehicle = original_vehicle

    def get_original_vehicle(self):
        return self.original_vehicle

    def get_original_vehicle_copy(self):
        return copy.deepcopy(self.original_vehicle)
