import gymnasium as gym
import highway_env

env = gym.make('merge-v0')
env.reset()
env = env.unwrapped

# 获取路网
network = env.road.network

# 遍历路网图结构：start_node -> end_node -> [lane0, lane1, ...]
for start_node in network.graph:
    for end_node in network.graph[start_node]:
        lanes = network.graph[start_node][end_node]  # 这是一个 Lane 对象列表
        for lane_index_in_list, lane in enumerate(lanes):
            lane_id = (start_node, end_node, lane_index_in_list)
            print(f"车道ID: {lane_id}, 类型: {type(lane).__name__}, 宽度: {lane.width}")