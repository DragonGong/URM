import yaml
from urm.train import train_model
from urm.config import Config

import matplotlib

matplotlib.use('TkAgg')  # 在导入 pyplot 前设置后端
# matplotlib.use('MacOSX')


def load_config_as_object(config_path="config/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


if __name__ == "__main__":
    config = load_config_as_object()
    train_model(config)
