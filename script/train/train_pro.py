import yaml
from urm.train import train_model
from urm.config import Config
import argparse
import matplotlib

matplotlib.use('TkAgg')  # 在导入 pyplot 前设置后端
# matplotlib.use('MacOSX')


def load_config_as_object(config_path="config/config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='训练模型的脚本，支持通过命令行指定配置文件路径')

    # 添加配置文件路径参数，-c 和 --config 都可使用，指定默认值
    parser.add_argument('-c', '--config',
                        type=str,
                        default="config/config.yaml",
                        help='配置文件 config.yaml 的路径（默认: config/config.yaml）')

    # 解析命令行参数
    args = parser.parse_args()

    # 加载配置并训练模型
    config = load_config_as_object(args.config)
    train_model(config)
