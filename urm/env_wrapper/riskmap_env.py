import cv2
import logging
import time
from matplotlib.colors import LinearSegmentedColormap
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame

from urm.config import Config
from urm.reward.reward_meta import RewardMeta
from urm.reward.state.ego_state import EgoState
from urm.reward.state.surrounding_state import SurroundingState
from urm.env_wrapper.env import Env
from urm.utils import Mode


class RiskMapEnv(Env):
    def __init__(self, env, config: Config, reward: RewardMeta, **kwargs):
        super().__init__(env, config, **kwargs)
        self.reward = reward
        self.last_acceleration = 0.0
        self.risk_map = None
        self.risk = 0

    def step(self, action):
        original_start = time.time()
        start = time.time()
        initial_lane_index  = None
        target_lane_index = None
        if hasattr(self.unwrapped, "controlled_vehicles"):
            c_v = self.unwrapped.controlled_vehicles[0]
            initial_lane_index = c_v.lane_index

        obs, base_line_reward, terminated, truncated, info = self.env.step(action)
        if hasattr(self.unwrapped, "controlled_vehicles"):
            c_v = self.unwrapped.controlled_vehicles[0]
            target_lane_index = c_v.target_lane_index

        logging.debug("\n\n\n")
        logging.debug(f"the baseline step time consuming is {time.time() - start}s")
        start = time.time()
        # if  self.config.training.render_mode:
        #     self.env.render()
        env = self.env.unwrapped

        ego_state = EgoState.from_vehicle(env.vehicle, env=self)
        surrounding_state = SurroundingState.from_road_vehicles(road_vehicles=env.road.vehicles,
                                                                exclude_vehicle=env.vehicle, env=self)

        logging.debug(f"the state transfer time consuming is {time.time() - start}s")
        start = time.time()

        risk = 0
        if (self.config.reward.version == 0 and (
                self.config.reward.baseline_reward_w == 1 and self.config.reward.custom_reward_w == 0)) \
                or (self.config.reward.version == 1 and (
                self.config.reward.baseline_reward_w == 1 and self.config.reward.risk_reward_w == 0)):
            reward = base_line_reward
        else:
            if initial_lane_index == target_lane_index:
                action = np.int64(1)
            reward, risk, self.risk_map = self.reward.reward(ego_state, surrounding_state, self, base_line_reward,
                                                             action)
            logging.debug(f"the reward calculation time is {time.time() - start}s")

        current_speed = env.vehicle.speed
        current_acceleration = env.vehicle.action["acceleration"] if hasattr(env.vehicle, 'action') else 0.0
        # jerk = |a_t - a_{t-1}|
        jerk = abs(current_acceleration - self.last_acceleration)
        self.last_acceleration = current_acceleration
        if "is_success" not in info:
            logging.debug("is_success is not in info")
            is_success = (
                    not info.get("crashed", False) and
                    terminated  # 只有正常结束才算成功（非 crash 导致的 terminated）
            )
        else:
            is_success = info.get("is_success", False)
        info.update({
            "speed": current_speed,
            "acceleration": current_acceleration,
            "jerk": jerk,
            "crashed": env.vehicle.crashed,
            "is_success": is_success,
            "on_road": env.vehicle.on_road,
            "risk": risk,
            "action": self.get_action_dict()[action]
        })
        self.risk = risk
        logging.debug(f"the step last for {time.time() - original_start}s")
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        base_render = self.env.render()
        if base_render is None or self.risk_map is None:
            return base_render

        if self.config.test_config.render_mode != "rgb_array":
            return base_render
        image = base_render.copy()
        original_image = base_render.copy()
        env = self.env.unwrapped
        vehicle = env.vehicle

        x_ego, y_ego = vehicle.position  # (x, y) in world frame
        theta = vehicle.heading  # radians, 0 = x-axis, positive = counter-clockwise

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])  # 旋转矩阵：局部 → 世界

        if not hasattr(env, 'viewer') or env.viewer is None:
            return image
        viewer = env.viewer
        sim_surface = viewer.sim_surface
        cmap = LinearSegmentedColormap.from_list(
            'green_to_red',
            [
                (0.0, (0, 1, 0)),  # Green
                (0.25, (1, 1, 0)),  # Yellow
                (0.5, (1, 0.647, 0)),  # Orange
                (0.75, (1, 0.27, 0)),  # Orange-red
                (1.0, (1, 0, 0))  # Red
            ],
            N=256
        )

        risk_avg = self.risk_map.finalize()  # shape: (ny, nx)
        mask = (self.risk_map.count > 0)

        for i in range(self.risk_map.ny):
            for j in range(self.risk_map.nx):
                if not mask[i, j]:
                    continue
                x_local = self.risk_map.x_min + (j + 0.5) * self.risk_map.cell_size
                y_local = self.risk_map.y_min + (i + 0.5) * self.risk_map.cell_size

                local_pos = np.array([x_local, y_local])
                world_pos = np.array([x_ego, y_ego]) + R @ local_pos

                try:
                    pixel_pos = sim_surface.vec2pix(world_pos)
                    px, py = int(pixel_pos[0]), int(pixel_pos[1])
                except Exception as e:
                    logging.error(f"vec2pix conversion failed: {e}")
                    continue

                H, W = image.shape[:2]
                if not (0 <= px < W and 0 <= py < H):
                    continue

                risk_val = float(np.clip(risk_avg[i, j], 0.0, 1.0))
                color_rgba = cmap(risk_val)  # (r, g, b, a) in [0,1]
                color_bgr = (np.array(color_rgba[:3]) * 255).astype(np.uint8)
                image = _draw_oriented_square(
                    image,
                    px=px,
                    py=py,
                    cell_size=self.risk_map.cell_size,
                    ppm=viewer.config["scaling"],
                    orientation=theta,
                    color_bgr=color_bgr
                )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 0,0 )
        thickness = 2
        text = f"Risk: {self.risk:.3f}"

        cv2.putText(image, text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        # combined = np.hstack((original_image, image))
        combined = np.vstack((image, original_image))
        return combined


def _draw_oriented_square(image, px, py, cell_size, ppm, orientation, color_bgr):
    side_length = int(cell_size * ppm)
    if side_length <= 0:
        return image

    half_side = side_length // 2
    points = np.array([[-half_side, -half_side],
                       [half_side, -half_side],
                       [half_side, half_side],
                       [-half_side, half_side]], dtype=np.float32)

    rotation_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
                                [np.sin(orientation), np.cos(orientation)]])

    rotated_points = np.dot(points, rotation_matrix.T)
    rotated_points[:, 0] += px
    rotated_points[:, 1] += py

    cv2.fillPoly(image, pts=[rotated_points.astype(np.int32)], color=color_bgr.tolist())

    return image
