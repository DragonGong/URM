import os
import imageio
from urm.test.tester import make_env, test_model
from urm.config.config import Config


def test_with_video(config, video_path="test_video.mp4"):
    """测试并保存视频"""
    env = make_env(config, render_mode="rgb_array")
    env = env  # 单环境，非 VecEnv

    model_path = config.test_config.model_path
    from stable_baselines3 import DQN, PPO
    algo = DQN if "dqn" in model_path.lower() else PPO
    model = algo.load(model_path)

    images = []
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        img = env.render()
        images.append(img)

    imageio.mimsave(video_path, images, fps=15)
    print(f"🎥 视频已保存至: {video_path}")
    env.close()
