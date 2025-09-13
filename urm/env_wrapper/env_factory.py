from urm.env_wrapper.baseline_env import BaselineEnv
from urm.env_wrapper.urm_env import URMHighwayEnv
from urm.env_wrapper.riskmap_env import RiskMapEnv  # 假设你有这个文件
from urm.reward.reward_meta import RewardMeta  # 如果 RiskMapEnv 用到
from urm.reward.riskmap_reward import RiskMapReward


def make_wrapped_env(env, config):
    """
    根据 config.env_wrapper 动态选择环境包装器
    """
    env_wrapper_type = getattr(config, 'env_wrapper', 'baseline').lower()

    if env_wrapper_type == "baseline":
        return BaselineEnv(env, config)
    elif env_wrapper_type == "urm":
        return URMHighwayEnv(env, config)
    elif env_wrapper_type == "riskmap":
        reward_instance = RiskMapReward(config)
        return RiskMapEnv(env, reward=reward_instance)
    else:
        raise ValueError(f"未知 env_wrapper 类型: {env_wrapper_type}, 支持: baseline, urm, riskmap")
