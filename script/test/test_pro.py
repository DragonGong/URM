import argparse
import logging

from urm import setup_logging_without_conf
from urm.config.config import Config
from urm.test import test_model


def load_config(config_path="config/config.yaml"):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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
    setup_logging_without_conf(log_dir='log', log_name_prefix="test", console_level=logging.INFO,
                               file_level=logging.DEBUG)

    config_dict = load_config(args.config)
    config = Config(config_dict)
    test_model(config)
