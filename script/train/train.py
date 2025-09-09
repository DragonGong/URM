import datetime
import os
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from urm.env_wrapper.urm_env import URMHighwayEnv
import highway_env


# 读取 YAML 配置
def load_config(config_path="config/train_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_env(config):
    env = gym.make(config['env_config']['env_id'])
    env_config = config['env_config']

    # 构建 IDMVehicle 配置（嵌套）
    idm_config = env_config.get('IDMVehicle', {})
    if idm_config:
        env_config['highway_env.vehicle.behavior.IDMVehicle'] = idm_config

    env.unwrapped.configure(env_config)
    env.reset()
    return URMHighwayEnv(env, config)


if __name__ == "__main__":
    # 加载配置
    config = load_config()

    # 创建环境
    env = DummyVecEnv([lambda: make_env(config)])

    # 从配置创建模型
    model_config = config['model_config']
    model = PPO(
        policy=model_config['policy'],
        env=env,
        verbose=model_config['verbose'],
        learning_rate=model_config['learning_rate'],
        n_steps=model_config['n_steps'],
        batch_size=model_config['batch_size'],
        n_epochs=model_config['n_epochs'],
        gamma=model_config['gamma'],
        device=model_config['device']
    )

    print("开始训练...")
    training_config = config['training']
    model.learn(total_timesteps=training_config['total_timesteps'])

    # 保存模型
    model_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_urm_highway"
    save_path = os.path.join(training_config['save_dir'], model_filename)
    os.makedirs(training_config['save_dir'], exist_ok=True)  # 确保目录存在
    model.save(save_path)

    print(f"模型已保存至: {save_path}")
