from urm.reward.state.car_state import CarState


class EgoState(CarState):
    def __init__(self, env, x=0.0, y=0.0, vx=0.0, vy=0.0, **kwargs):
        super().__init__(env=env, x=x, y=y, vx=vx, vy=vy, **kwargs)

    @classmethod
    def from_vehicle(cls, vehicle, **kwargs):
        """
        从环境中的 vehicle 对象（如 env.vehicle）构造 EgoState
        假设 vehicle 有 .position (x,y) 和 .velocity (vx,vy)
        """
        pos = vehicle.position
        vel = vehicle.velocity
        return cls(x=pos[0], y=pos[1], vx=vel[0], vy=vel[1], **kwargs)

    def __repr__(self):
        return f"EgoState(x={self.x:.2f}, y={self.y:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f})"
