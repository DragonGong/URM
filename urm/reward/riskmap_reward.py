from typing import Optional

from urm.config import Config
from urm.reward.reward_meta import RewardMeta
from urm.reward.riskmap.risk_map import RiskMap
from urm.reward.riskmap.riskmap_manager import RiskMapManager
from urm.reward.state.ego_state import EgoState
from urm.reward.state.interface import EnvInterface
from urm.reward.state.state import State
from urm.reward.state.surrounding_state import SurroundingState
from urm.reward.trajectory.behavior import BehaviorFactory
from urm.reward.trajectory.trajectory_generator import TrajectoryGenerator
from urm.reward.trajectory.prediction import *
from urm.reward.utils.riskmap_visualizer import RiskMapVisualizer


class RiskMapReward(RewardMeta):
    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.riskmap_manager: Optional[RiskMapManager] = None
        self.behavior_factory = BehaviorFactory(config.reward.behavior_configs)
        self.prediction_model = create_model_from_config(self.config)
        self.behaviors = self.behavior_factory.get_all_behaviors_by_config()
        if self.config.reward.visualize:
            self.visualizer = RiskMapVisualizer(title="Training RiskMap", plt_show=config.reward.plt_show)
        else:
            self.visualizer = None

    def reward(self, ego_state: EgoState, surrounding_states: SurroundingState, env_condition: EnvInterface,
               baseline_reward):
        print(f"baseline reward is {baseline_reward}")
        if self.riskmap_manager is None:
            urm_risk = baseline_reward
        else:
            riskmap_total: RiskMap = self.riskmap_manager.sum_all()
            if self.visualizer is not None:
                vis_data = riskmap_total.get_visualization_data()
                self.visualizer.update(vis_data)
            self.riskmap_manager.plot_all()
            # riskmap_total.plot_promax((ego_state.vehicle_size.length, ego_state.vehicle_size.width))
            riskmap_total.plot_pro(finalize=False)
            custom = riskmap_total.get_risk_for_car(ego_state, self.riskmap_manager.world_to_local)
            urm_risk = self.urm_risk(
                custom_risk=custom,
                baseline=baseline_reward)
            print(f"custom_risk is {custom}")
        self.riskmap_manager_create(ego_state=ego_state, surrounding_states=surrounding_states,
                                    env_condition=env_condition)
        return urm_risk

    def urm_risk_magnify(self, custom_risk, baseline):
        return

    def urm_risk(self, custom_risk, baseline):
        return self.config.reward.baseline_reward_w * baseline + (
                1 - self.config.reward.baseline_reward_w) * custom_risk

    def riskmap_manager_create(self, ego_state: EgoState, surrounding_states: SurroundingState,
                               env_condition: EnvInterface):
        global_state = State(env=env_condition)
        generator = TrajectoryGenerator(ego_state, surrounding_states, env_condition=global_state,
                                        behaviors=self.behaviors,
                                        prediction_model=self.prediction_model, config=self.config)
        traj = generator.generate_right(
            self.config.reward.step_num,
            self.config.reward.duration)
        generator.set_risk_backpropagation(traj)
        if self.visualizer is not None:
            traj.visualize()
        self.riskmap_manager = RiskMapManager(config=self.config.reward, trajtree=traj)
        self.riskmap_manager.assign_risk_with_vehicle()
