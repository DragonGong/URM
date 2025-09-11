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
            self.tensorboard_log = kwargs.get("tensorboard_log", "highway_dqn/")

    class RewardConfig:
        def __init__(self, **kwargs):
            self.desired_speed = kwargs.get("desired_speed", 20)
            self.r_safe_w = kwargs.get("r_safe_w", 0.7)
            self.r_speed_w = kwargs.get("r_speed_w", 0.2)
            self.r_lateral_w = kwargs.get("r_lateral_w", 0.1)
            self.v2r_w = kwargs.get("v2r_w", 0.1)
            self.baseline_reward_w = kwargs.get("baseline_reward_w", 0)

            # 公式参数：
            self.discount_factor_max = kwargs.get("discount_factor_max", 0.8)
            self.velocity_unit = kwargs.get("velocity_unit", 20)
            self.discount_factor_min = kwargs.get("discount_factor_min", 0.2)

            self.risk_max_for_tree = kwargs.get("risk_max_for_tree", 1)  # 如果碰撞直接设的值

    class TrainingConfig:
        def __init__(self, **kwargs):
            self.total_timesteps = kwargs.get("total_timesteps", 10000)
            self.save_dir = kwargs.get("save_dir", "./agent")

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
