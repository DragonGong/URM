from typing import List, Dict


class Config:
    class EnvConfig:
        def __init__(self, **kwargs):
            # 必需字段
            self.env_id = kwargs.get("env_id", "highway-fast-v0")
            self.lanes_count = kwargs.get("lanes_count", 4)
            self.vehicles_count = kwargs.get("vehicles_count", 50)
            self.duration = kwargs.get("duration", 400)
            self.simulation_frequency = kwargs.get("simulation_frequency", 15)
            self.policy_frequency = kwargs.get("policy_frequency", 1)
            self.initial_spacing = kwargs.get("initial_spacing", 2)

            # observation 配置
            self.observation = kwargs.get("observation", {"type": "Kinematics"})

            self.other_vehicles_type = kwargs.get("other_vehicles_type", "highway_env.vehicle.behavior.IDMVehicle")
            # vehicles_density 可能被用在环境中生成车辆
            self.vehicles_density = kwargs.get("vehicles_density", 1.0)

            # action 配置
            self.action = kwargs.get("action", {"type": "DiscreteMetaAction"})

            # Reward settings
            self.collision_reward = kwargs.get("collision_reward", -1)
            self.reward_speed_range = kwargs.get("reward_speed_range", [20, 30])
            # 注释掉的 lane_change_reward 不设置默认值，除非需要
            self.centering_position = kwargs.get("centering_position", [0.3, 0.5])

            # Rendering
            self.screen_width = kwargs.get("screen_width", 600)
            self.screen_height = kwargs.get("screen_height", 150)
            self.scaling = kwargs.get("scaling", 5.5)
            self.show_trajectories = kwargs.get("show_trajectories", False)
            self.render_agent = kwargs.get("render_agent", True)
            self.offscreen_rendering = kwargs.get("offscreen_rendering", False)

    class ModelConfig:
        # 强化学习模型
        def __init__(self, **kwargs):
            self.algorithm = kwargs.get("algorithm", "DQN")
            self.policy = kwargs.get("policy", "MlpPolicy")
            self.policy_kwargs = kwargs.get("policy_kwargs", {"net_arch": [256, 256]})
            self.learning_rate = kwargs.get("learning_rate", 0.0005)
            self.buffer_size = kwargs.get("buffer_size", 15000)
            self.learning_starts = kwargs.get("learning_starts", 200)
            self.batch_size = kwargs.get("batch_size", 32)
            self.gamma = kwargs.get("gamma", 0.8)
            self.train_freq = kwargs.get("train_freq", 1)
            self.gradient_steps = kwargs.get("gradient_steps", 1)
            self.target_update_interval = kwargs.get("target_update_interval", 50)
            self.verbose = kwargs.get("verbose", 1)
            self.tensorboard_log = kwargs.get("tensorboard_log", None)

    class RewardConfig:

        class FittingModelConfigs:
            class Polynomial:
                def __init__(self, **kwargs):
                    pass

            def __init__(self, **kwargs):
                self.polynomial_config = self.Polynomial(**kwargs.get("polynomial", {}))
                self.interval_duration = kwargs.get("interval_duration", 0.1)

        class PredictionModelConfigs:
            class LinearModelConfig:
                def __init__(self, **kwargs):
                    pass

            class IDMModelConfig:
                def __init__(self, **kwargs):
                    pass

            def __init__(self, **kwargs):
                self.linear_model_config = self.LinearModelConfig(**kwargs.get("linear_model_config", {}))
                self.idm_model_config = self.IDMModelConfig(**kwargs.get("idm_model_config", {}))

        class BehaviorSequenceConfig:
            """单个时间步的行为配置"""

            def __init__(self, lateral: str, longitudinal: str):
                self.lateral = lateral
                self.longitudinal = longitudinal

        class ScenarioConfig:
            """一个完整场景的行为序列配置"""

            def __init__(self, name: str, description: str, sequence: List[Dict[str, str]]):
                self.name = name
                self.description = description
                self.sequence = [
                    Config.RewardConfig.BehaviorSequenceConfig(**step)
                    for step in sequence
                ]

        class BehaviorConfigs:
            def __init__(self, **kwargs):
                self.behavior_configs = kwargs.get("behaviors_list", [])
                scenarios = kwargs.get("scenarios", [])
                self.scenarios = [
                    Config.RewardConfig.ScenarioConfig(**scenario)
                    for scenario in scenarios
                ]

        class RiskMapConfig:
            def __init__(self, **kwargs):
                self.cell_size = kwargs.get("cell_size", 0.5)

        def __init__(self, **kwargs):
            # 轨迹树建立参数
            self.step_num = kwargs.get("step_num", 3)
            self.duration = kwargs.get("duration", 1)  # 单位是秒

            # general 参数
            self.desired_speed = kwargs.get("desired_speed", 20)
            self.r_safe_w = kwargs.get("r_safe_w", 0.7)
            self.r_speed_w = kwargs.get("r_speed_w", 0.2)
            self.r_lateral_w = kwargs.get("r_lateral_w", 0.1)
            self.v2r_w = kwargs.get("v2r_w", 0.1)
            self.baseline_reward_w = kwargs.get("baseline_reward_w", 0)
            self.custom_reward_w = kwargs.get("custom_reward_w", 0)

            # 公式参数：
            self.discount_factor_max = kwargs.get("discount_factor_max", 0.8)
            self.velocity_unit = kwargs.get("velocity_unit", 20)
            self.discount_factor_min = kwargs.get("discount_factor_min", 0.2)

            self.risk_max_for_tree = kwargs.get("risk_max_for_tree", 1)  # 如果碰撞直接设的值

            # 预测参数：
            self.surrounding_radius = kwargs.get("surrounding_radius", 100)
            # 预测模型：
            self.prediction_model = kwargs.get("prediction_model", "linear_model")

            self.prediction_model_configs = self.PredictionModelConfigs(**kwargs.get("prediction_model_configs", {}))

            # 拟合模型：
            self.fitting_model = kwargs.get("fitting_model", "polynomial")
            self.fitting_model_configs = self.FittingModelConfigs(**kwargs.get("fitting_model_config", {}))

            # 行为列表
            self.behavior_configs = self.BehaviorConfigs(**kwargs.get("behavior_configs", {}))

            # riskmap config
            self.riskmap_config = self.RiskMapConfig(**kwargs.get("riskmap_config", {}))

            # 可视化
            self.visualize = kwargs.get("visualize", False)
            self.plt_show = kwargs.get("plt_show", False)

    class TrainingConfig:
        def __init__(self, **kwargs):
            self.total_timesteps = kwargs.get("total_timesteps", 10000)
            self.save_dir = kwargs.get("save_dir", "./agent")
            self.render_mode = kwargs.get("render_mode", None)  # 默认不渲染
            self.n_eval_episodes = kwargs.get("n_eval_episodes", 20)
            self.eval_freq = kwargs.get("eval_freq", 1000)

    class TestConfig:
        def __init__(self, **kwargs):
            self.model_path = kwargs.get("model_path", "./agent/default_model")
            self.render_mode = kwargs.get("render_mode", None)  # 默认不渲染
            self.test_episodes = kwargs.get("test_episodes", 1)

    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}

        # 初始化各个子配置
        self.env_config = self.EnvConfig(**config_dict.get("env_config", {}))
        self.model_config = self.ModelConfig(**config_dict.get("model_config", {}))
        self.reward = self.RewardConfig(**config_dict.get("reward", {}))
        self.training = self.TrainingConfig(**config_dict.get("training", {}))
        self.test_config = self.TestConfig(**config_dict.get("test_config", {}))

        self.env_wrapper = config_dict.get("env_wrapper", "baseline")

    def to_dict(self):
        """将整个 Config 对象转换为嵌套字典,会过滤掉私有变量，_ 开头的这种"""

        def _obj_to_dict(obj):
            if hasattr(obj, "__dict__"):
                return {k: _obj_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif isinstance(obj, dict):
                return {k: _obj_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_obj_to_dict(x) for x in obj)
            else:
                return obj

        return _obj_to_dict(self)

    def __repr__(self):
        return (f"<Config(\n"
                f"  env_config={self.env_config.__dict__},\n"
                f"  model_config={self.model_config.__dict__},\n"
                f"  reward={self.reward.__dict__},\n"
                f"  training={self.training.__dict__},\n"
                f"  test_config={self.test_config.__dict__}\n"
                f")>")
