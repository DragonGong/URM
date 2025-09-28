import time
from copy import deepcopy
from typing import List, Optional, Tuple, Dict

from urm.config import Config
from urm.reward.state.car_state import CarState
from urm.reward.state.ego_state import EgoState
from urm.reward.state.region2d import Region2D
from urm.reward.state.state import State
from urm.reward.state.surrounding_state import SurroundingState
from urm.reward.trajectory.behavior.behavioral_combination import BehavioralCombination
from urm.reward.trajectory.behavior.behaviors import Behavior
from urm.reward.trajectory.highway_env_state import HighwayState
from urm.reward.trajectory.prediction.model import Model
from urm.reward.trajectory.traj import TrajNode, TrajEdge
from urm.reward.trajectory.traj_tree import TrajTree
from urm.reward.trajectory.fitting import *


class TrajectoryGenerator:
    def __init__(self, ego_state: EgoState, surrounding_states: SurroundingState, env_condition: State,
                 behaviors: Dict[str, List['BehavioralCombination']],
                 prediction_model: Model, config: Config):
        self.env_condition = env_condition.env_condition
        self.config: Optional[Config] = config
        self.surrounding_states = surrounding_states
        self.ego_state = ego_state
        self.behaviors = behaviors
        self.prediction_model = prediction_model
        self.prediction_result: [[Region2D]] = []
        start_time = time.time()
        self.predict_collision_region()
        print(f"predict_collision_region time is {time.time()-start_time}")
        self.fitting_algorithm: Fitting = create_fitting_from_config(self.config)

    def fitting_edge(self, edge: TrajEdge):
        self.fitting_algorithm.fit_edge_by_node(edge)

    def fitting_edge_frenet(self, edge: TrajEdge):
        self.fitting_algorithm.fit_edge_by_node_frenet(edge, self.env_condition)

    def set_edge_risk(self, edge: TrajEdge):
        last_risk = edge.node_end.risk
        for i in reversed(range(0, len(edge.discrete_points))):
            risk = last_risk.get_value() * max(self.config.reward.discount_factor_max - edge.discrete_points[
                i].velocity.magnitude / self.config.reward.velocity_unit, self.config.reward.discount_factor_min)
            edge.discrete_points[i].set_risk_value(t=edge.discrete_points[i].get_time(), risk=risk,
                                                   speed=edge.discrete_points[i].velocity.magnitude)

    def predict_collision_region(self):
        """
        Generate the temporal and spatial regions of conflicts among surrounding vehicles
        :return:
        """
        step_num = self.config.reward.step_num
        duration = self.config.reward.duration

        for t in range(step_num):
            result = []
            for car in self.surrounding_states:
                # s = time.time()
                region = self.prediction_model.predict_region(car, (t + 1) * duration, width=car.vehicle_size.width,
                                                              length=car.vehicle_size.length)
                # print(f"each car prediction time for 1 s is {time.time()-s}")
                result.append(region)
            self.prediction_result.append(result)

    def generate_right(self, step_nums=3, duration=1):
        root_node = TrajNode.from_car_state(self.ego_state)
        # traj_tree = self.traj_tree_generated_by_behaviors(root_node, self.ego_state, self.behaviors, step_nums,
        #                                                   duration)
        start_time = time.time()
        traj_tree = self.traj_tree_generated_by_behavior_collection(root_node=root_node, ego_state=self.ego_state,
                                                                    behaviors=self.behaviors, duration=duration)
        print(f"traj_tree_generated_by_behavior_collection duration is {time.time()-start_time}")
        start_time = time.time()
        # traj_tree.visualize_plot_nb(show_direction=False)
        traj_tree = self.traj_tree_cut(traj_tree)
        print(f"traj_tree_cut duration is {time.time()-start_time}")
        # traj_tree.visualize_plot_nb(show_direction=False)
        return traj_tree

    def traj_tree_generated_by_behavior_collection(
            self,
            root_node: TrajNode,
            ego_state: CarState,
            behaviors: Dict[str, List['BehavioralCombination']],
            duration: int,
    ) -> TrajTree:
        """
        根据行为序列列表，直接构建轨迹树。
        每个行为序列 behaviors[i] = [b0, b1, b2, ...] 表示从 root 开始，连续执行的行为。
        构建时允许节点重合（不合并、不查重）。
        优化版：构建时直接维护当前子树引用，避免递归查找。
        """
        if not behaviors:
            return TrajTree.from_node(root_node)

        root_tree = TrajTree.from_node(root_node)

        for behavior_sequence in behaviors.values():
            current_node = root_node
            current_state = ego_state
            current_tree = root_tree  # 当前所在的子树（初始为根树）

            for step_idx, behavior in enumerate(behavior_sequence):
                # 计算目标状态
                target_car_state = behavior.target_state(
                    initial_position=current_state.position_cls,
                    state=HighwayState.from_carstate(current_state, duration)
                )

                # 创建新节点
                new_node = TrajNode.from_car_state(target_car_state)
                new_node.set_timestep(step_idx + 1)

                # 创建边并拟合
                edge = TrajEdge(current_node, new_node)
                assert isinstance(behavior, BehavioralCombination), "behavior is not BehavioralCombination"
                edge.action = (behavior.longitudinal.behavior_type, behavior.lateral.behavior_type)
                self.fitting_edge_frenet(edge)

                # 创建新子树（叶子）
                new_subtree = TrajTree.from_node(new_node)
                # 挂接到当前子树
                current_tree.add_child(edge, new_subtree)

                # 推进到下一步
                current_node = new_node
                current_state = target_car_state
                current_tree = new_subtree  # 下一步的“当前子树”就是刚创建的这个

        return root_tree

    def traj_tree_generated_by_behaviors(self, root_node: TrajNode, ego_state: CarState, behaviors: List[Behavior],
                                         step_nums,
                                         duration: int, time_step=0) -> TrajTree:
        if time_step == step_nums:
            return TrajTree.from_node(root_node)
        tree_list = []
        for b in behaviors:
            car_state = b.target_state(ego_state.position_cls, HighwayState.from_carstate(ego_state, duration))
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
        对轨迹树进行碰撞剪枝（修正版，保留引用一致性）：
        - 返回剪枝后的新树（不修改原树）
        """

        def _prune_tree(tree: TrajTree) -> Optional[Tuple[TrajTree, Dict]]:
            """
            返回 (pruned_tree, mapping)
            mapping: 原节点 -> 新节点（只包含这个子树中出现的原节点）
            """
            # 复制根节点并建立映射
            new_root = deepcopy(tree.root)
            mapping = {tree.root: new_root}

            safe_children = []

            for edge, child_tree in tree.iter_children():
                # 采样并检测碰撞（使用原始 edge 的采样）
                sampled_points: List[TrajNode] = edge.sample(num_points=10)
                collision_detected = any(
                    self.judge_conventional_collision(point.car_state)
                    for point in sampled_points
                )

                if collision_detected:
                    continue  # 该边碰撞，剪掉整条子树

                # 递归剪枝子树，得到子树的新副本以及子映射
                pruned_result = _prune_tree(child_tree)
                if pruned_result is None:
                    # 子树被完全剪掉
                    continue

                pruned_child_tree, child_mapping = pruned_result

                # 合并映射：这样 mapping 中包含当前子树中所有已复制的节点
                mapping.update(child_mapping)

                # 复制边（深拷贝边对象），然后修正其 node 引用为新对象
                new_edge = deepcopy(edge)

                # 常见的可能保存节点引用的属性名（防御式处理）
                candidate_attrs = ("node_begin", "node_end", "begin", "end", "from_node", "to_node")
                for attr in candidate_attrs:
                    if hasattr(new_edge, attr):
                        val = getattr(new_edge, attr)
                        # 如果该属性引用了原节点对象，将其替换为新节点引用
                        if val in mapping:
                            setattr(new_edge, attr, mapping[val])

                # 兼容性保障：确保 node_begin/node_end 至少指向父/子新节点
                # edge.node_begin 通常是 tree.root，edge.node_end 通常是 child_tree.root
                if hasattr(edge, "node_begin") and edge.node_begin in mapping:
                    new_edge.node_begin = mapping[edge.node_begin]
                if hasattr(edge, "node_end") and edge.node_end in mapping:
                    new_edge.node_end = mapping[edge.node_end]

                # 将（新边，新子树）绑定保存
                safe_children.append((new_edge, pruned_child_tree))

            # 构造返回的新子树并返回映射
            if len(safe_children) == 0:
                return TrajTree.from_node(new_root), mapping
            else:
                return TrajTree(root=new_root, children=safe_children), mapping

        result = _prune_tree(traj_tree)
        if result is None:
            # 全部被剪掉，返回仅含根节点的副本以保证不返回 None
            return TrajTree.from_node(deepcopy(traj_tree.root))
        pruned_tree, _ = result
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
                    # todo：直接根据这个整条边是否被撞，然后来直接赋值整条边的risk，不合理，需要更加精细化
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

    def judge_conventional_collision(self, car_state: CarState):
        return not self.env_condition.judge_on_road(x=car_state.x, y=car_state.y, length=car_state.vehicle_size.length,
                                                    width=car_state.vehicle_size.width,
                                                    direction=car_state.velocity.direction)

    def judge_surrounding_collision(self, node: TrajNode):
        # 这个函数需要在多个环节调用，为了效率只预测一遍
        # 与时间有关，所以是 节点 为输入（节点输入包含时间）
        assert self.prediction_result is not None, "prediction result is None!"
        regions: [Region2D] = self.prediction_result[int(node.get_time()) - 1]
        for region in regions:
            if region.contains((node.x, node.y)):
                return True
        return False
