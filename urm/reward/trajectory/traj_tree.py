from typing import List, Optional, Iterator
from matplotlib.patches import FancyArrowPatch

from urm.reward.state.utils.position import Position
from urm.reward.trajectory.traj import TrajNode, TrajEdge


class TrajTree:
    def __init__(
            self,
            root: TrajNode,
            children_edges: Optional[List['TrajEdge']] = None,
            children_trees: Optional[List['TrajTree']] = None
    ):
        """
        轨迹树节点：包含一个根节点，以及通过边连接的子树列表。
        每条边 TrajEdge 从当前 root 指向子树的 root。
        """
        self.root = root
        self.children_edges = children_edges or []
        self.children_trees = children_trees or []

        # 确保边和子树数量一致
        if len(self.children_edges) != len(self.children_trees):
            raise ValueError("children_edges 和 children_trees 长度必须一致")

    @classmethod
    def from_node(cls, node: TrajNode) -> 'TrajTree':
        """仅包含根节点的叶子树"""
        return cls(root=node)

    def add_child(self, edge: TrajEdge, subtree: 'TrajTree'):
        """添加一个子树，通过指定的边连接"""
        if edge.node_begin != self.root:
            raise ValueError("边的起点必须是当前树的根节点")
        self.children_edges.append(edge)
        self.children_trees.append(subtree)

    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点（无子树）"""
        return len(self.children_trees) == 0

    @property
    def total_edges(self) -> int:
        """递归计算树中所有边的数量"""
        count = len(self.children_edges)
        for child in self.children_trees:
            count += child.total_edges
        return count

    @property
    def total_nodes(self) -> int:
        """递归计算树中所有节点的数量"""
        count = 1  # 当前根节点
        for child in self.children_trees:
            count += child.total_nodes
        return count

    def dfs_iter(self) -> Iterator['TrajTree']:
        """深度优先遍历所有子树（包括自身）"""
        yield self
        for child_tree in self.children_trees:
            yield from child_tree.dfs_iter()

    def sample_all_edges(self, num_points: int = 10) -> List[Position]:
        """
        对树中所有边进行采样，返回所有采样点（不包括节点，仅边上的插值点）
        """
        points = []
        for edge in self.children_edges:
            edge_points = edge.sample(num_points)
            # 可选：去掉起点（因为是root，可能重复），保留中间点和终点
            # points.extend(edge_points[1:])  # 去掉起点
            points.extend(edge_points)  # 保留所有点
        for child in self.children_trees:
            points.extend(child.sample_all_edges(num_points))
        return points

    def get_all_nodes(self) -> List[TrajNode]:
        """递归获取所有节点"""
        nodes = [self.root]
        for child in self.children_trees:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_all_edges(self) -> List[TrajEdge]:
        """递归获取所有边"""
        edges = self.children_edges[:]
        for child in self.children_trees:
            edges.extend(child.get_all_edges())
        return edges

    def total_length(self) -> float:
        """递归计算树中所有边的直线距离总和"""
        length = sum(edge.length for edge in self.children_edges)
        for child in self.children_trees:
            length += child.total_length()
        return length

    def __repr__(self) -> str:
        child_reprs = ", ".join([f"→{child.root}" for child in self.children_trees])
        return f"TrajTree({self.root}, children=[{child_reprs}])"

    def visualize(self, indent: int = 0) -> str:
        """简单文本可视化树结构"""
        lines = []
        indent_str = "  " * indent
        lines.append(f"{indent_str}└─ {self.root}")
        for i, child in enumerate(self.children_trees):
            edge = self.children_edges[i]
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
        使用 matplotlib 可视化整棵轨迹树

        参数:
            ax: matplotlib axis，若为 None 则新建
            show_nodes: 是否显示节点
            show_edges: 是否显示边
            show_sample_points: 是否显示边上的采样点
            sample_points_num: 每条边采样点数
            node_color: 节点颜色
            edge_color: 边颜色
            sample_point_color: 采样点颜色
            node_size: 节点大小
            title: 图标题
            show_grid: 是否显示网格
            show_legend: 是否显示图例
            show_direction: 是否用箭头显示边方向
            dpi: 图像分辨率
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
            created_fig = True
        else:
            created_fig = False

        all_nodes = self.get_all_nodes()
        all_edges = self.get_all_edges()

        # 绘制节点
        if show_nodes:
            x_nodes = [node.x for node in all_nodes]
            y_nodes = [node.y for node in all_nodes]
            ax.scatter(x_nodes, y_nodes, color=node_color, s=node_size, zorder=5,
                       label='TrajNode' if show_legend else "")
            # 可选：标注节点序号
            for i, node in enumerate(all_nodes):
                ax.text(node.x + 0.02, node.y + 0.02, str(i + 1), fontsize=9, color='darkred')

        # 绘制边
        if show_edges:
            for edge in all_edges:
                sampled = edge.sample(sample_points_num)
                x_vals = [p.x for p in sampled]
                y_vals = [p.y for p in sampled]

                if show_direction and len(x_vals) > 1:
                    # 画带箭头的线（从起点到终点方向）
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
                    # 也可以只画一个从起点指向终点的大箭头（更简洁）：
                    # ax.annotate('', xy=(x_vals[-1], y_vals[-1]), xytext=(x_vals[0], y_vals[0]),
                    #             arrowprops=dict(arrowstyle='->', color=edge_color, lw=2))
                else:
                    ax.plot(x_vals, y_vals, color=edge_color, linewidth=2, zorder=3,
                            label='TrajEdge' if show_legend and edge == all_edges[0] else "")

                # 绘制采样点（可选）
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
