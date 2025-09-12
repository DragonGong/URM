from copy import deepcopy
from typing import List, Optional

from urm.config import Config
from urm.reward.state.car_state import CarState
from urm.reward.state.ego_state import EgoState
from urm.reward.state.region2d import Region2D
from urm.reward.state.state import State
from urm.reward.state.surrounding_state import SurroundingState
from urm.reward.state.utils.position import Position
from urm.reward.trajectory.behavior.behaviors import Behavior
from urm.reward.trajectory.highway_env_state import HighwayState
from urm.reward.trajectory.prediction.model import Model
from urm.reward.trajectory.risk import Risk
from urm.reward.trajectory.traj import TrajNode, TrajEdge
from urm.reward.trajectory.traj_tree import TrajTree
from urm.reward.trajectory.fitting import *


class TrajectoryGenerator:
    def __init__(self, ego_state: EgoState, surrounding_states: SurroundingState, env_condition: State,
                 behaviors: List[Behavior],
                 prediction_model: Model, config: Config):
        self.env_condition = env_condition.env_condition
        self.config: Optional[Config] = config
        self.surrounding_states = surrounding_states
        self.ego_state = ego_state
        self.behaviors = behaviors
        self.prediction_model = prediction_model
        self.prediction_result: [[Region2D]] = None
        self.predict_collision_region()
        self.fitting_algorithm: Fitting = create_fitting_from_config(self.config)

    def fitting_edge(self, edge: TrajEdge):
        self.fitting_algorithm.fit_edge_by_node(edge)

    def set_edge_risk(self, edge: TrajEdge):
        last_risk = edge.node_end.risk
        for i in reversed(range(0, len(edge.discrete_points))):
            risk = last_risk.get_value() * max(self.config.reward.discount_factor_max - edge.discrete_points[
                i].velocity.magnitude / self.config.reward.velocity_unit, self.config.reward.discount_factor_min)
            edge.discrete_points[i].risk.set_value(t=edge.discrete_points[i].get_time(), risk=risk,
                                                   speed=edge.discrete_points[i].velocity.magnitude)

    def predict_collision_region(self):
        """
        Generate the temporal and spatial regions of conflicts among surrounding vehicles
        :return:
        """
        step_num = self.config.reward.step_num
        duration = self.config.reward.duration

        for time in range(step_num):
            result = []
            for car in self.surrounding_states:
                region = self.prediction_model.predict_region(car, (time + 1) * duration, width=car.vehicle_size.width,
                                                              length=car.vehicle_size.length)
                result.append(region)
            self.prediction_result.append(result)

    def generate_right(self, step_nums=3, duration=1):
        root_node = TrajNode.from_car_state(self.ego_state)
        traj_tree = self.traj_tree_generated_by_behaviors(root_node, self.ego_state, self.behaviors, step_nums,
                                                          duration)
        traj_tree = self.traj_tree_cut(traj_tree)
        return traj_tree

    def traj_tree_generated_by_behaviors(self, root_node: TrajNode, ego_state: CarState, behaviors: List[Behavior],
                                         step_nums,
                                         duration: int, time_step=0) -> TrajTree:
        if time_step == step_nums:
            return TrajTree.from_node(root_node)
        tree_list = []
        for b in behaviors:
            car_state = b.target_state(ego_state.position, HighwayState.from_carstate(ego_state, duration))
            # 建节点的时候，就会把速度引进去
            node = TrajNode.from_car_state(car_state)
            node.set_timestep(time_step + 1)
            tree_list.append(
                self.traj_tree_generated_by_behaviors(node, car_state, behaviors, step_nums, duration, time_step + 1))
        tree = TrajTree.from_node(root_node)
        for child_tree in tree_list:
            edge = TrajEdge(root_node, child_tree.root)
            self.fitting_edge(edge)
            tree.add_child(edge, child_tree)
        return tree

    def traj_tree_cut(self, traj_tree: TrajTree) -> TrajTree:
        """
        对轨迹树进行碰撞剪枝：
        - 遍历每条边，采样点调用 judge_collision
        - 若任意采样点碰撞 → 删除该边及子树
        - 否则保留
        - 返回剪枝后的新树（不修改原树）
        """

        def _prune_tree(tree: TrajTree) -> Optional[TrajTree]:
            # 创建当前节点的副本（深拷贝根节点）
            new_root = deepcopy(tree.root)

            # 存储安全的子边和子树（绑定在一起）
            safe_children = []

            # ✅ 直接遍历绑定的 (edge, child_tree) 元组，语义清晰，绝对安全
            for edge, child_tree in tree.iter_children():
                # 采样边上的点（默认10个点）
                sampled_points = edge.sample(num_points=10)

                # 检查所有采样点是否碰撞
                collision_detected = any(
                    self.judge_conventional_collision(point)
                    for point in sampled_points
                )

                if not collision_detected:
                    # 安全：递归剪枝子树
                    pruned_child = _prune_tree(child_tree)
                    if pruned_child is not None:  # 子树可能被完全剪掉
                        # 深拷贝边（避免修改原边）
                        new_edge = deepcopy(edge)
                        new_edge.node_begin = new_root  # 修正起点为新根
                        # ✅ 直接添加绑定元组
                        safe_children.append((new_edge, pruned_child))

            # 如果没有任何子树保留，返回叶子节点
            if len(safe_children) == 0:
                return TrajTree.from_node(new_root)
            else:
                # ✅ 使用新的构造方式（只传 children 列表）
                return TrajTree(root=new_root, children=safe_children)

        # 开始剪枝
        pruned_tree = _prune_tree(traj_tree)
        if pruned_tree is None:
            return TrajTree.from_node(deepcopy(traj_tree.root))
        return pruned_tree

    def set_risk_backpropagation(self, tree: TrajTree):
        root = tree.root
        if self.judge_surrounding_collision(root):
            self.set_tree_risk_unification(tree, self.config.reward.risk_max_for_tree)
            return
        else:
            for edge in tree.children_edges:
                sample_nodes = edge.sample()
                collision = False
                for sample_node in sample_nodes:
                    assert isinstance(sample_node, TrajNode), "sample node is not type of TrajNode!"
                    if self.judge_surrounding_collision(sample_node):
                        collision = True
                        break
                if collision:
                    # todo：直接根据这个整条边是否被撞，然后来直接负值整条边的risk，不合理，需要更加精细化
                    self.set_tree_risk_unification(tree.get_subtree_by_edge(edge), self.config.reward.risk_max_for_tree)
                else:
                    self.set_risk_backpropagation(tree.get_subtree_by_edge(edge))

            if len(tree.children_trees) == 0:
                root.set_risk_value(0, root.velocity.magnitude)
                return
            else:
                risk_value = 0.0
                for edge in tree.children_edges:
                    assert len(edge.discrete_points) != 0, "edge.discrete_points == 0 when set_risk_backpropagation"
                    self.set_edge_risk(edge)
                    risk_value += edge.discrete_points[0].risk.get_value()
                root.set_risk_value(risk_value / len(tree.children_edges))
        return

    def set_tree_risk_unification(self, tree: TrajTree, value):
        if tree is None:
            return
        tree.root.set_risk_value(value, tree.root.velocity.magnitude)
        for child_edge in tree.children_edges:
            assert (child_edge.discrete_points is not None) and (
                    len(child_edge.discrete_points) > 0), "child edge discrete points is nil!"
            for p in child_edge.discrete_points:
                p.set_risk_value(risk=value)
        for child_tree in tree.children_trees:
            self.set_tree_risk_unification(child_tree, value)

    def judge_conventional_collision(self, position: Position):
        return self.env_condition.judge_match_road(position)

    def judge_surrounding_collision(self, node: TrajNode):
        # 这个函数需要在多个环节调用，为了效率只预测一遍
        # 与时间有关，所以是 节点 为输入（节点输入包含时间）
        assert self.prediction_result is not None, "prediction result is None!"
        regions: [Region2D] = self.prediction_result[node.get_time() - 1]
        for region in regions:
            if region.contains(node.x, node.y):
                return True
        return False
