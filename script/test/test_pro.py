from urm.config.config import Config
from urm.test import test_model


def load_config(config_path="config/config.yaml"):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config_dict = load_config()
    config = Config(config_dict)
    test_model(config)
