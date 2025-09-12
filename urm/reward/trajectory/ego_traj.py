#!/usr/bin/env python3
"""
Sampling-based trajectory planning in Frenet frame with Behavior Tree.
Supports integration with gymnasium environments.
"""
from enum import Enum
from typing import List, Optional, Tuple
from urm.reward.trajectory.highway_env_state import HighwayState as State
from urm.reward.trajectory.behavior.behaviors import LateralKeep, LateralLeft, LateralRight, LongitudinalCruise, \
    SoftAcceleration, HardAcceleration, SoftDeceleration, HardDeceleration


# =============================
# Behavior Tree Nodes  by lqy
# =============================
class BehaviorNode:
    """Tree node storing a (lateral, longitudinal) behavior."""

    def __init__(self, lateral: str, longitudinal: str):
        self.behavior: Tuple[str, str] = (lateral, longitudinal)
        self.children: List["BehaviorNode"] = []
        self.children_behavior: List[Tuple[str, str]] = []

    def add_child(self, node: "BehaviorNode") -> None:
        self.children.append(node)
        self.children_behavior.append(node.behavior)

    def find(self, behavior: Tuple[str, str]) -> Optional["BehaviorNode"]:
        """Find child node by behavior tuple."""
        if behavior in self.children_behavior:
            idx = self.children_behavior.index(behavior)
            return self.children[idx]
        return None


class LateralBehaviorType(str, Enum):
    KEEP = "Keep"
    LEFT = "Left"
    RIGHT = "Right"


class LongitudinalBehaviorType(str, Enum):
    CRUISE = "Cruise"
    SOFT_ACC = "SoftAcc"
    HARD_ACC = "HardAcc"
    SOFT_DEC = "SoftDec"
    HARD_DEC = "HardDec"


class BehaviorTree:
    """Behavior Tree managing all possible driving behavior paths."""

    def __init__(self):
        self.root = BehaviorNode("Root", "Root")
        self.lateral_behaviors = {
            LateralBehaviorType.KEEP: LateralKeep(),
            LateralBehaviorType.LEFT: LateralLeft(),
            LateralBehaviorType.RIGHT: LateralRight(),
        }
        self.longitudinal_behaviors = {
            LongitudinalBehaviorType.CRUISE: LongitudinalCruise(),
            LongitudinalBehaviorType.SOFT_ACC: SoftAcceleration(),
            LongitudinalBehaviorType.HARD_ACC: HardAcceleration(),
            LongitudinalBehaviorType.SOFT_DEC: SoftDeceleration(),
            LongitudinalBehaviorType.HARD_DEC: HardDeceleration(),
        }

        # Define available paths
        self._build_paths()

    def _build_paths(self):
        """Construct predefined behavior paths with 3-step temporal sequence."""

        # 路径1: 向左变道 → 保持车道并开始加速 → 继续加速
        # - 第1步: 执行向左变道（Lateral: Left），纵向保持巡航（避免同时剧烈操作）
        # - 第2步: 已完成或接近完成变道，保持当前车道（Keep），开始软加速（SoftAcc）
        # - 第3步: 稳定在目标车道，继续软加速以达到期望速度
        self.add_path([
            BehaviorNode(LateralBehaviorType.LEFT, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_ACC),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_ACC)
        ])

        # 路径2: 向右变道 → 保持车道并开始加速 → 继续加速
        # - 第1步: 执行向右变道（Lateral: Right），纵向保持巡航（确保横向稳定性）
        # - 第2步: 变道过程中或完成后保持车道（Keep），启动软加速（SoftAcc）
        # - 第3步: 稳定在新车道，持续软加速以提升速度
        self.add_path([
            BehaviorNode(LateralBehaviorType.RIGHT, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_ACC),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_ACC)
        ])

        # 路径3: 持续巡航 → 继续巡航 → 向左变道并轻微加速
        # - 第1步: 保持当前车道巡航（Keep），无变道需求
        # - 第2步: 仍保持车道巡航，系统开始准备变道决策
        # - 第3步: 执行向左变道（Left），同时轻微加速（SoftAcc）以匹配目标车道速度
        self.add_path([
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.LEFT, LongitudinalBehaviorType.SOFT_ACC)
        ])

        # 路径4: 持续巡航 → 继续巡航 → 向右变道并轻微加速
        # - 第1步: 保持车道巡航，当前无动作
        # - 第2步: 继续保持车道，系统评估右变道时机
        # - 第3步: 执行向右变道（Right），同时轻微加速（SoftAcc）以适应新车道流速
        self.add_path([
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.RIGHT, LongitudinalBehaviorType.SOFT_ACC)
        ])

        # 路径5: 持续巡航 → 持续巡航 → 持续巡航
        # - 第1步: 保持车道（Keep），纵向巡航（Cruise）—— 当前状态稳定
        # - 第2步: 无变道或速度变化需求，继续保持巡航
        # - 第3步: 长期稳定行驶策略，适用于自由流交通场景
        self.add_path([
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.CRUISE),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.CRUISE)
        ])

        # 路径6: 平稳加速 → 继续平稳加速 → 持续平稳加速
        # - 第1步: 保持车道（Keep），开始软加速（SoftAcc）—— 如驶出匝道后提速
        # - 第2步: 继续在当前车道内平稳加速
        # - 第3步: 持续加速至目标速度，准备进入巡航状态
        self.add_path([
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_ACC),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_ACC),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_ACC)
        ])

        # 路径7: 平稳减速 → 继续平稳减速 → 持续平稳减速
        # - 第1步: 保持车道（Keep），开始软减速（SoftDec）—— 如前方车流减速
        # - 第2步: 在当前车道内继续平稳减速
        # - 第3步: 持续减速，为停车或低速通行做准备
        self.add_path([
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_DEC),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_DEC),
            BehaviorNode(LateralBehaviorType.KEEP, LongitudinalBehaviorType.SOFT_DEC)
        ])

    def add_path(self, behavior_path: List[BehaviorNode]) -> None:
        """Add a new path of behaviors to the tree."""
        search_node = self.root
        for node in behavior_path:
            child_node = search_node.find(node.behavior)
            if child_node is None:
                search_node.add_child(node)
                search_node = node
            else:
                search_node = child_node

    def get_next_behaviors(self, parent_action: Tuple[float, float], state: State,
                           path: List[Tuple[str, str]]):
        """Given a path of behaviors, return possible next (offset, velocity)."""
        search_node = self.root
        for behavior in path:
            search_node = search_node.find(behavior)
            if search_node is None:
                raise ValueError(f"Path {path} not found in tree.")

        behaviors, actions = [], []
        parent_d, parent_v = parent_action
        for lat, lon in search_node.children_behavior:
            d = self.lateral_behaviors[lat].target_offset(parent_d, state)
            v = self.longitudinal_behaviors[lon].target_velocity(parent_v, state)
            behaviors.append((lat, lon))
            actions.append((d, v))
        return behaviors, actions

    def print_paths(self) -> None:
        """Print all behavior paths in the tree."""
        results: List[str] = []
        self._recursive_collect(self.root, results, "")
        for path in results:
            print(path)

    def _recursive_collect(self, node: BehaviorNode, results: List[str], prefix: str):
        if not node.children:
            results.append(prefix)
        else:
            for child in node.children:
                self._recursive_collect(child, results, prefix + str(child.behavior) + "->")


if __name__ == "__main__":
    tree = BehaviorTree()
    tree.print_paths()
