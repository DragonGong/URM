import gymnasium as gym
import highway_env
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 创建环境
env = gym.make('roundabout-v0')
env.reset()
env = env.unwrapped
network = env.road.network

# 设置绘图
plt.figure(figsize=(14, 10))
plt.title("Highway-env Road Network Visualization (with Node Labels)", fontsize=16)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis('equal')  # 保持比例
plt.grid(True, linestyle='--', alpha=0.5)

# 用于记录已绘制的节点位置，避免重复标注
node_positions = {}

# 遍历路网
for start_node in network.graph:
    for end_node in network.graph[start_node]:
        lanes = network.graph[start_node][end_node]
        for lane_index_in_list, lane in enumerate(lanes):
            lane_id = (start_node, end_node, lane_index_in_list)

            # 生成车道中心线点（采样）
            s_values = np.linspace(0, getattr(lane, 'length', 50) if hasattr(lane, 'length') else 50, 100)
            x_coords = []
            y_coords = []
            for s in s_values:
                pos = lane.position(s, 0)  # d=0 表示中心线
                x_coords.append(pos[0])
                y_coords.append(pos[1])

            # 设置颜色和样式
            lane_type = type(lane).__name__
            color = 'blue' if lane_type == 'StraightLane' else 'red' if lane_type == 'CircularLane' else 'green'
            linestyle = '-' if lane_index_in_list == 0 else '--' if lane_index_in_list == 1 else ':'

            # 绘制车道中心线
            plt.plot(x_coords, y_coords,
                     color=color,
                     linestyle=linestyle,
                     linewidth=2,
                     label=f"{lane_id} ({lane_type})" if lane_index_in_list == 0 and start_node ==
                                                         list(network.graph.keys())[0] else "")

            # 获取起点和终点的世界坐标
            start_pos = lane.position(0, 0)
            end_pos = lane.position(s_values[-1], 0)

            # 记录节点位置（取第一次出现的位置）
            if start_node not in node_positions:
                node_positions[start_node] = start_pos
            if end_node not in node_positions:
                node_positions[end_node] = end_pos

            # 在车道起点附近标注 lane_id（只标注一次，避免重叠）
            if lane_index_in_list == 0:  # 只在每组第一条车道标注
                mid_idx = len(x_coords) // 4  # 在1/4处标注，避免堵在起点
                plt.text(x_coords[mid_idx], y_coords[mid_idx],
                         f"{lane_id}",
                         fontsize=8,
                         ha='center',
                         va='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# 标注所有节点名称
for node_name, pos in node_positions.items():
    plt.scatter(pos[0], pos[1], s=100, c='black', zorder=5)
    plt.text(pos[0], pos[1] + 1,
             node_name,
             fontsize=12,
             ha='center',
             va='bottom',
             weight='bold',
             bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", edgecolor="black"))

# 图例（去重）
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.show()
