from urm.reward.state.car_state import CarState
from urm.reward.state.state import State
from typing import List, Optional


class SurroundingState(State):
    def __init__(self, cars: Optional[List['CarState']] = None, **kwargs):
        super().__init__(**kwargs)
        self.cars = cars if cars is not None else []  # 存储 CarState 对象的列表

    @classmethod
    def from_road_vehicles(cls, road_vehicles, exclude_vehicle=None, **kwargs):
        """
        从 env.road.vehicles 构造 SurroundingState，可选排除某辆车（如 ego）
        """
        cars = []
        for v in road_vehicles:
            if v is exclude_vehicle:
                continue
            pos = v.position
            vel = v.velocity
            cars.append(CarState(x=pos[0], y=pos[1], vx=vel[0], vy=vel[1]))
        return cls(cars=cars, **kwargs)

    def add_car(self, car: 'CarState'):
        """添加一辆周围车辆"""
        self.cars.append(car)

    def remove_car(self, car: 'CarState'):
        """移除指定车辆（按对象引用）"""
        if car in self.cars:
            self.cars.remove(car)

    def remove_car_by_index(self, index: int):
        """根据索引移除车辆"""
        if 0 <= index < len(self.cars):
            del self.cars[index]

    def get_car(self, index: int) -> Optional['CarState']:
        """根据索引获取车辆，越界返回 None"""
        return self.cars[index] if 0 <= index < len(self.cars) else None

    def get_cars_in_radius(self, center_x: float, center_y: float, radius: float) -> List['CarState']:
        """获取指定圆半径范围内的所有车辆"""
        return [
            car for car in self.cars
            if ((car.x - center_x) ** 2 + (car.y - center_y) ** 2) <= radius ** 2
        ]

    def get_cars_in_front(self, ego_x: float, ego_heading: float, distance: float) -> List['CarState']:
        """
        获取在自车前方一定距离内的车辆（简化模型，假设 heading 是 x 轴方向）
        更复杂的实现可加入 yaw 角度计算
        """
        # 简化：假设自车朝向 x 正方向
        return [
            car for car in self.cars
            if car.x > ego_x and (car.x - ego_x) <= distance
        ]

    def clear(self):
        """清空所有车辆"""
        self.cars.clear()

    def __len__(self):
        """支持 len(surrounding_state)"""
        return len(self.cars)

    def __iter__(self):
        """支持 for car in surrounding_state"""
        return iter(self.cars)

    def __repr__(self):
        return f"SurroundingState(cars={len(self.cars)} vehicles)"

    @property
    def car_count(self) -> int:
        """只读属性：周围车辆数量"""
        return len(self.cars)

    def to_list(self) -> List['CarState']:
        """返回车辆列表副本（避免外部直接修改内部状态）"""
        return self.cars.copy()
