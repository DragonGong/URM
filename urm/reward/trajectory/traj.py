from copy import deepcopy

from urm.reward.state.car_state import CarState
from urm.reward.state.utils.position import Position
from typing import List, Optional, Callable, Iterator, Union, Tuple

from urm.reward.state.utils.velocity import Velocity
from urm.reward.trajectory.risk import Risk
from urm.reward.trajectory.behavior import BehaviorName


class TrajNode(Position):
    def __init__(self, x: float, y: float, **kwargs):
        super().__init__(x, y)
        self.risk: Optional[Risk] = kwargs.get('risk', None)
        self._time: float = 0  # 建树的时候就会赋值
        self.velocity: Optional[Velocity] = kwargs.get('velocity', None)
        self._car_state: Optional[CarState] = kwargs.get('car_state', None)

    def get_time(self) -> float:
        return self._time

    def set_velocity(self, velocity: Velocity):
        self.velocity = velocity

    @property
    def car_state(self) -> CarState:
        assert self._car_state is not None, "cat_state is fucking None!"
        return self._car_state

    def set_car_state(self, car_state: CarState):
        self._car_state = car_state

    @classmethod
    def from_car_state(cls, state: CarState):
        """
        车的状态中的速度和轨迹点的是不一样的地址
        :param state:
        :return:
        """
        node = cls.from_position(Position.from_tuple(state.position))
        node.set_velocity(state.velocity)
        node._car_state = deepcopy(state)
        return node

    @classmethod
    def from_position(cls, position: Position) -> 'TrajNode':
        return cls(position.x, position.y)

    def set_timestep(self, timestep: float):
        self._time = timestep

    def judge_risk_initial_nature(self) -> bool:
        if self.risk is not None:
            return False
        else:
            return True

    def set_risk_value(self, risk: float, speed: Optional[float] = None, t: float = -1):
        if self.risk is None:
            self.risk = Risk()
        if speed is None:
            speed = self.velocity.magnitude
        if t == -1:
            self.risk.set_value(self._time, risk, speed)
        else:
            self.risk.set_value(t, risk, speed)

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:
        return f"TrajNode({self.x:.3f}, {self.y:.3f})"

    def __eq__(self, other) -> bool:
        return self is other


class TrajEdge:
    def __init__(
            self,
            node_begin: TrajNode,
            node_end: TrajNode,
    ):
        self.node_begin = node_begin
        self.node_end = node_end
        self.discrete_points: Optional[List[TrajNode]] = None
        self.fitting_algorithm = None
        self.action: Optional[Tuple[
            BehaviorName, BehaviorName]] = None  # behavior.longitudinal.behavior_type, behavior.lateral.behavior_type

    def set_discrete_points(self, node_list: List[TrajNode]):
        self.discrete_points = node_list

    @property
    def discrete_points_reference(self):
        return self.discrete_points

    @property
    def length(self) -> float:
        """边的直线距离"""
        return self.node_begin.distance_to(self.node_end)

    def sample(self, num_points: int = 10) -> Union[List[Position], List[TrajNode]]:
        """
        采样该边上的点。
        如果有 fitting_algorithm，使用它生成点；
        否则默认线性插值。
        """
        if self.discrete_points is not None:
            return self.discrete_points
        else:
            # 默认：线性插值
            points = []
            for i in range(num_points + 1):
                t = i / num_points
                x = self.node_begin.x * (1 - t) + self.node_end.x * t
                y = self.node_begin.y * (1 - t) + self.node_end.y * t
                points.append(Position(x, y))
            return points

    def __repr__(self) -> str:
        algo_name = self.fitting_algorithm.__name__ if self.fitting_algorithm else "linear"
        return f"TrajEdge({self.node_begin} → {self.node_end}, algo={algo_name})"


# abandoned
class Traj:
    # 单个轨迹
    def __init__(
            self,
            nodes: List[TrajNode],
            default_fitting_algorithm: Optional[Callable[[TrajEdge], List[Position]]] = None
    ):
        if len(nodes) < 2:
            raise ValueError("Trajectory must have at least 2 nodes.")

        self.nodes = nodes
        self.default_fitting_algorithm = default_fitting_algorithm
        # 自动构建边
        self.edges: List[TrajEdge] = []
        for i in range(len(nodes) - 1):
            edge = TrajEdge(
                node_begin=nodes[i],
                node_end=nodes[i + 1],
            )
            self.edges.append(edge)

    @property
    def length(self) -> float:
        """轨迹总长度（各边长度之和）"""
        return sum(edge.length for edge in self.edges)

    def sample(self, points_per_edge: int = 10) -> List[Position]:
        """
        对整条轨迹采样，每条边采样 points_per_edge 个点
        返回平坦的点列表（包含所有边的采样点）
        """
        sampled_points = []
        for edge in self.edges:
            edge_points = edge.sample(num_points=points_per_edge)
            sampled_points.extend(edge_points[:-1])  # 避免重复端点（最后一个点留给下一条边开头）
        # 添加最后一个点
        sampled_points.append(self.nodes[-1])
        return sampled_points

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[TrajNode]:
        return iter(self.nodes)

    def __getitem__(self, index: int) -> TrajNode:
        return self.nodes[index]

    def append_node(self, node: TrajNode):
        """在轨迹末尾添加一个节点，并创建新边"""
        if not self.nodes:
            self.nodes = [node]
            return
        last_node = self.nodes[-1]
        new_edge = TrajEdge(last_node, node)
        self.nodes.append(node)
        self.edges.append(new_edge)

    def insert_node(self, index: int, node: TrajNode):
        """在指定位置插入节点，并更新相邻边"""
        if index < 0 or index > len(self.nodes):
            raise IndexError("Index out of range")

        if index == len(self.nodes):
            self.append_node(node)
            return

        # 插入节点
        self.nodes.insert(index, node)

        # 重建 edges
        self.edges = []
        for i in range(len(self.nodes) - 1):
            edge = TrajEdge(
                node_begin=self.nodes[i],
                node_end=self.nodes[i + 1],
            )
            self.edges.append(edge)

    def __repr__(self) -> str:
        return f"Traj({len(self.nodes)} nodes, {len(self.edges)} edges, length={self.length:.2f})"
