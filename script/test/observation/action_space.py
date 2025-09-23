import gymnasium as gym
import highway_env

# 创建环境
env = gym.make("intersection-v1")

# 重置环境（必须，否则 action_type 可能未初始化）
env.reset()

# 获取动作映射字典
actions_dict = env.unwrapped.action_type.actions  # 或 env.action_type.actions

# 打印动作列表
print("动作列表：")
for action_id, action_name in actions_dict.items():
    print(f"动作 {action_id}: {action_name}")