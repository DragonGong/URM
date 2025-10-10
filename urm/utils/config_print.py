import pprint


def config_to_dict(config):
    """
    递归将 Config 对象（或其子对象）转换为普通 dict。
    支持 dataclass、有 __dict__ 的类、dict、list 等。
    """
    if hasattr(config, "__dict__"):
        # 是一个类实例（如 Config、RewardConfig 等）
        result = {}
        for key, value in config.__dict__.items():
            if key.startswith('_'):  # 跳过私有属性
                continue
            result[key] = config_to_dict(value)
        return result
    elif isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, (list, tuple)):
        return [config_to_dict(item) for item in config]
    else:
        # 基本类型（int, float, str, bool, None）直接返回
        return config


def write_config_to_file(config, filepath):
    """
    将 Config 对象以美观的格式写入 txt 文件。
    """
    config_dict = config_to_dict(config)

    with open(filepath, 'w', encoding='utf-8') as f:
        # 方法 A：使用 pprint（推荐，Python 风格，易读）
        pprint.pprint(config_dict, stream=f, width=100, sort_dicts=False)

