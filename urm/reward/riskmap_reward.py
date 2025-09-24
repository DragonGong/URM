from collections import Counter
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from urm.config import Config
from urm.reward.reward_meta import RewardMeta
from urm.reward.riskmap.risk_map import RiskMap
from urm.reward.riskmap.riskmap_manager import RiskMapManager
from urm.reward.state.ego_state import EgoState
from urm.reward.state.interface import EnvInterface
from urm.reward.state.state import State
from urm.reward.state.surrounding_state import SurroundingState
from urm.reward.trajectory.behavior import BehaviorFactory, BehaviorName
from urm.reward.trajectory.traj import TrajNode
from urm.reward.trajectory.traj_tree import TrajTree
from urm.reward.trajectory.trajectory_generator import TrajectoryGenerator
from urm.reward.trajectory.prediction import *
from urm.reward.utils.riskmap_visualizer import RiskMapVisualizer


class RiskMapReward(RewardMeta):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.riskmap_manager: Optional[RiskMapManager] = None
        self.behavior_factory = BehaviorFactory(config.reward.behavior_configs)
        self.prediction_model = create_model_from_config(self.config)
        self.behaviors = self.behavior_factory.get_all_scenarios_as_combinations()
        if self.config.reward.visualize:
            self.visualizer = RiskMapVisualizer(title="Training RiskMap", plt_show=config.reward.plt_show)
        else:
            self.visualizer = None

    def reward(self, ego_state: EgoState, surrounding_states: SurroundingState, env_condition: EnvInterface,
               baseline_reward, action):
        print(f"baseline reward is {baseline_reward}")
        if self.riskmap_manager is None:
            urm_risk = baseline_reward
        else:
            riskmap_total: RiskMap = self.riskmap_manager.sum_all()
            if self.visualizer is not None:
                vis_data = riskmap_total.get_visualization_data()
                self.visualizer.update(vis_data)
            # self.riskmap_manager.plot_all()
            riskmap_total.plot_pro()
            # custom = riskmap_total.get_risk_for_car(ego_state, self.riskmap_manager.world_to_local)
            action_dict = env_condition.get_action_dict()
            print(f"action is {action_dict[int(action)]}")
            traj_nodes = self.get_tree_nodes_by_action(action, self.riskmap_manager.trajtree, action_dict)
            print(f"traj node num is {len(traj_nodes)}")
            # plot_traj_nodes(traj_nodes)
            plot_traj_nodes_with_counts(traj_nodes)
            if traj_nodes is None or len(traj_nodes) <= 0:
                custom = 0
            else:
                risk_all, cell_count, riskmap_mask = self.riskmap_manager.get_risk_by_tree(
                    traj_nodes=traj_nodes,
                    risk_map=riskmap_total)
                riskmap_mask.plot_pro(block=True)
                print(f"cell count is {cell_count}")
                custom = risk_all / cell_count
            urm_risk = self.urm_risk(
                custom_risk=custom,
                baseline=baseline_reward)
            print(f"custom_risk is {custom}")
        self.riskmap_manager_create(ego_state=ego_state, surrounding_states=surrounding_states,
                                    env_condition=env_condition)
        return urm_risk

    def urm_risk(self, custom_risk, baseline):
        return self.config.reward.baseline_reward_w * baseline + (
                1 - self.config.reward.baseline_reward_w) * custom_risk

    def riskmap_manager_create(self, ego_state: EgoState, surrounding_states: SurroundingState,
                               env_condition: EnvInterface):
        global_state = State(env=env_condition)
        generator = TrajectoryGenerator(ego_state, surrounding_states, env_condition=global_state,
                                        behaviors=self.behaviors,
                                        prediction_model=self.prediction_model, config=self.config)
        traj = generator.generate_right(
            self.config.reward.step_num,
            self.config.reward.duration)
        generator.set_risk_backpropagation(traj)
        if self.visualizer is not None:
            traj.visualize()
        self.riskmap_manager = RiskMapManager(config=self.config.reward, trajtree=traj)
        self.riskmap_manager.assign_risk_with_vehicle()

    def get_tree_nodes_by_action(self, action, trajtree: TrajTree, action_dict):
        action_str = action_dict[int(action)]
        behavior_combination_list = map_action_str_to_behavior(action_str)
        all_nodes = []
        for edge, child_tree in trajtree.iter_children():
            if edge.action is None:
                continue
            if edge.action in behavior_combination_list:
                assert edge.discrete_points is not None, "edge.discrete_points is None"
                # print(f"edge.discrete point num is {len(edge.discrete_points)}. ")
                all_nodes.extend(edge.discrete_points)
                all_nodes.extend(child_tree.get_all_nodes_with_edge_nodes())
        return all_nodes


def plot_traj_nodes(traj_nodes: List[TrajNode], block=False):
    """
    根据给定的轨迹节点列表绘制图形。

    参数:
        traj_nodes: 一个包含轨迹节点的列表。假定每个节点都有 'x' 和 'y' 属性。
    """
    # 提取所有点的 x 和 y 坐标
    x_coords = [node.x for node in traj_nodes]
    y_coords = [node.y for node in traj_nodes]

    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, color='blue', label='Trajectory Nodes')

    # 添加标题和标签
    plt.title('Trajectory Nodes Visualization')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 展示图形
    plt.show(block=block)


def plot_traj_nodes_with_counts(traj_nodes: List[TrajNode], show_labels: bool = True, block=False):
    """
    绘制 traj_nodes 中所有节点的位置，并显示每个位置的节点数量。

    参数:
        traj_nodes: 轨迹节点列表，每个节点需有 .x 和 .y 属性
        show_labels: 是否在点旁边显示数量（仅当 count > 1 时显示）
    """
    if not traj_nodes:
        print("警告：输入的 traj_nodes 为空，无法绘图。")
        return

    # 提取坐标并统计频次
    coords = [(round(node.x, 6), round(node.y, 6)) for node in traj_nodes]  # 避免浮点误差
    counter = Counter(coords)

    # 去重后的坐标和对应频次
    unique_coords = list(counter.keys())
    counts = [counter[coord] for coord in unique_coords]
    x_vals = [c[0] for c in unique_coords]
    y_vals = [c[1] for c in unique_coords]

    # 点的大小：基础大小 + 与 count 成正比（避免太小或太大）
    base_size = 30
    sizes = [base_size * count for count in counts]

    # 创建图形
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_vals, y_vals, s=sizes, c=counts, cmap='viridis', alpha=0.7, edgecolors='k')

    # 可选：添加数量标签（仅当 count > 1）
    if show_labels:
        for (x, y), count in zip(unique_coords, counts):
            if count > 1:
                plt.text(x, y, str(count), fontsize=9, ha='center', va='center', color='white',
                         fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, boxstyle='circle,pad=0.1'))

    plt.title('Trajectory Nodes with Overlap Counts')
    plt.xlabel('X (world coordinates)')
    plt.ylabel('Y (world coordinates)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(scatter, label='Number of Nodes at Position')

    # 阻塞直到窗口关闭
    plt.show(block=block)


def map_action_str_to_behavior(action_str: str) -> List[Tuple[BehaviorName, BehaviorName]]:
    action_str = action_str.strip().upper()
    if action_str == "LANE_LEFT":
        return [(BehaviorName.CRUISE, BehaviorName.LATERAL_LEFT)]
    elif action_str == "LANE_RIGHT":
        return [(BehaviorName.CRUISE, BehaviorName.LATERAL_RIGHT)]
    elif action_str == "IDLE":
        return [(BehaviorName.CRUISE, BehaviorName.LATERAL_KEEP)]
    elif action_str == "FASTER":
        return [
            (BehaviorName.SOFT_ACCEL, BehaviorName.LATERAL_KEEP),
            (BehaviorName.HARD_ACCEL, BehaviorName.LATERAL_KEEP)
        ]
    elif action_str == "SLOWER":
        return [
            (BehaviorName.SOFT_DECEL, BehaviorName.LATERAL_KEEP),
            (BehaviorName.HARD_DECEL, BehaviorName.LATERAL_KEEP)
        ]
    else:
        raise ValueError(
            f"不支持的动作字符串: '{action_str}'。"
            f" 支持的动作: ['LANE_LEFT', 'LANE_RIGHT', 'IDLE', 'FASTER', 'SLOWER']"
        )
