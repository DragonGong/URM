from matplotlib.patches import FancyArrowPatch
from urm.reward.state.utils.position import Position
from urm.reward.trajectory.traj import TrajNode, TrajEdge
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass


class TrajTree:
    def __init__(
            self,
            root: TrajNode,
            children: Optional[List[Tuple['TrajEdge', 'TrajTree']]] = None
    ):
        """
        轨迹树节点：包含一个根节点，以及通过边连接的子树列表。
        每条边 TrajEdge 从当前 root 指向子树的 root。
        """
        self.root = root
        self._children = children or []  # 类型: List[(edge, subtree)]

    @classmethod
    def from_node(cls, node: TrajNode) -> 'TrajTree':
        """仅包含根节点的叶子树"""
        return cls(root=node)

    def add_child(self, edge: TrajEdge, subtree: 'TrajTree'):
        """添加一个子树，通过指定的边连接"""
        if edge.node_begin != self.root:
            raise ValueError("边的起点必须是当前树的根节点")
        self._children.append((edge, subtree))

    @property
    def children_edges(self) -> List[TrajEdge]:
        """获取所有子边（只读视图）"""
        return [edge for edge, _ in self._children]

    @property
    def children_trees(self) -> List['TrajTree']:
        """获取所有子树（只读视图）"""
        return [subtree for _, subtree in self._children]

    def get_subtree_by_edge(self, edge: 'TrajEdge') -> 'TrajTree':
        """
        通过边查找对应的子树（语义匹配起点终点）
        """
        for e, subtree in self._children:
            if (e.node_begin == edge.node_begin and
                    e.node_end == edge.node_end):
                return subtree
        raise ValueError(f"边 {edge} 不是当前树 {self.root} 的直接子边")

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def total_edges(self) -> int:
        count = len(self._children)
        for _, child in self._children:
            count += child.total_edges
        return count

    @property
    def total_nodes(self) -> int:
        count = 1
        for _, child in self._children:
            count += child.total_nodes
        return count

    def dfs_iter(self) -> Iterator['TrajTree']:
        yield self
        for _, child in self._children:
            yield from child.dfs_iter()

    def sample_all_edges(self, num_points: int = 10) -> List[Position]:
        points = []
        for edge, _ in self._children:
            points.extend(edge.sample(num_points))
        for _, child in self._children:
            points.extend(child.sample_all_edges(num_points))
        return points

    def get_all_nodes(self) -> List[TrajNode]:
        nodes = [self.root]
        for _, child in self._children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_all_nodes_with_edge_nodes(self) -> List[TrajNode]:
        nodes = [self.root]
        assert self._children is not None, "self._children is none"
        for edge, child_tree in self._children:
            nodes.extend(edge.sample())
            nodes.extend(child_tree.get_all_nodes_with_edge_nodes())
        return nodes

    def get_all_edges(self) -> List[TrajEdge]:
        edges = [edge for edge, _ in self._children]
        for _, child in self._children:
            edges.extend(child.get_all_edges())
        return edges

    def total_length(self) -> float:
        length = sum(edge.length for edge, _ in self._children)
        for _, child in self._children:
            length += child.total_length()
        return length

    def iter_children(self) -> Iterator[Tuple['TrajEdge', 'TrajTree']]:
        """
        迭代器：安全遍历所有子边和对应子树的绑定对。
        推荐用于遍历子结构，避免直接访问内部 _children。
        """
        yield from self._children

    def __repr__(self) -> str:
        child_reprs = ", ".join([f"→{subtree.root}" for _, subtree in self._children])
        return f"TrajTree({self.root}, children=[{child_reprs}])"

    def visualize(self, indent: int = 0) -> str:
        """简单文本可视化树结构"""
        lines = []
        indent_str = "  " * indent
        lines.append(f"{indent_str}└─ {self.root}")
        for edge, child in self._children:  # ✅ 安全遍历绑定关系
            algo_name = edge.fitting_algorithm.__name__ if edge.fitting_algorithm else "linear"
            lines.append(f"{indent_str}  ├─ Edge({algo_name}) →")
            lines.append(child.visualize(indent + 2))
        return "\n".join(lines)

    def visualize_plot(
            self,
            ax=None,
            show_nodes=True,
            show_edges=True,
            show_sample_points=False,
            sample_points_num=10,
            node_color='red',
            edge_color='blue',
            sample_point_color='green',
            node_size=50,
            title="Trajectory Tree Visualization",
            show_grid=True,
            show_legend=True,
            show_direction=True,
            dpi=100
    ):
        """
        使用 matplotlib 可视化整棵轨迹树（已适配新结构，安全可靠）
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
            created_fig = True
        else:
            created_fig = False

        all_nodes = self.get_all_nodes()  # ✅ 已重构，安全
        all_edges = self.get_all_edges()  # ✅ 已重构，安全

        # 绘制节点
        if show_nodes:
            x_nodes = [node.x for node in all_nodes]
            y_nodes = [node.y for node in all_nodes]
            ax.scatter(x_nodes, y_nodes, color=node_color, s=node_size, zorder=5,
                       label='TrajNode' if show_legend else "")
            for i, node in enumerate(all_nodes):
                ax.text(node.x + 0.02, node.y + 0.02, str(i + 1), fontsize=9, color='darkred')

        # 绘制边
        if show_edges:
            for edge in all_edges:
                sampled = edge.sample(sample_points_num)
                x_vals = [p.x for p in sampled]
                y_vals = [p.y for p in sampled]

                if show_direction and len(x_vals) > 1:
                    for i in range(len(x_vals) - 1):
                        arrow = FancyArrowPatch(
                            (x_vals[i], y_vals[i]),
                            (x_vals[i + 1], y_vals[i + 1]),
                            arrowstyle='->,head_width=1.5,head_length=2',
                            color=edge_color,
                            alpha=0.8,
                            linewidth=2,
                            mutation_scale=10,
                            zorder=3
                        )
                        ax.add_patch(arrow)
                else:
                    ax.plot(x_vals, y_vals, color=edge_color, linewidth=2, zorder=3,
                            label='TrajEdge' if show_legend and edge == all_edges[0] else "")

                if show_sample_points:
                    ax.scatter(x_vals[1:-1], y_vals[1:-1], color=sample_point_color, s=15, zorder=4,
                               label='Sample Point' if show_legend and edge == all_edges[0] else "")

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.6)
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        ax.autoscale()
        if created_fig:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')  # 在导入 pyplot 前设置后端
    # 创建节点
    n1 = TrajNode(0, 0)
    n2 = TrajNode(2, 2)
    n3 = TrajNode(3, 0)
    n4 = TrajNode(1, 3)

    # 创建边（默认线性）
    e1 = TrajEdge(n1, n2)
    e2 = TrajEdge(n2, n3)
    e3 = TrajEdge(n1, n4)

    # 构建树
    leaf3 = TrajTree.from_node(n3)
    leaf4 = TrajTree.from_node(n4)

    tree2 = TrajTree(n2)
    tree2.add_child(e2, leaf3)

    root_tree = TrajTree(n1)
    root_tree.add_child(e1, tree2)
    root_tree.add_child(e3, leaf4)

    # 可视化！
    root_tree.visualize_plot(
        show_sample_points=True,
        sample_points_num=20,
        show_direction=True,
        title="Trajectory Tree with Sampled Points & Direction Arrows"
    )
