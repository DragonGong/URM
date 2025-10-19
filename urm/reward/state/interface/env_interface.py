from typing import Protocol, Any, Tuple, Union, runtime_checkable, Optional
import numpy as np


@runtime_checkable
class EnvInterface(Protocol):
    def get_action_dict(self):
        ...

    def _extract_xy(self, pos: Any) -> Tuple[float, float]:
        ...

    def judge_match_road(
            self,
            pos: Union[Tuple[float, float], np.ndarray, dict],
            margin: float = 0.5
    ) -> bool:
        ...

    def judge_on_road(self, x, y, length, width, direction, margin: float = 0.2) -> bool:
        ...

    def world_to_lane_local(
            self,
            x: float,
            y: float,
            lane_id: Tuple[str, str, int]
    ) -> Tuple[float, float]:
        ...

    def lane_local_to_world(
            self,
            lane_id: Tuple[str, str, int],
            longitudinal: float,
            lateral: float
    ) -> Tuple[float, float]:
        ...

    def get_current_road_segment(
            self,
            x: float,
            y: float,
            return_lane_id: bool = False
    ) -> Union[Tuple[str, str], Tuple[str, str, Tuple[str, str, int]]]:
        ...

    def get_frenet_velocity(self, x: float, y: float, vx: float, vy: float,
                            land_id: Optional[Tuple[str, str, int]] = None) -> Tuple[
        float, float, Tuple[str, str, int]]:
        ...

    def frenet_velocity_to_cartesian(
            self,
            x: float,
            y: float,
            v_lon: float,
            v_lat: float,
            lane_id: Tuple[str, str, int],
    ) -> tuple[np.ndarray[tuple[int, ...], Any], np.ndarray[tuple[int, ...], Any]]:
        ...

    def get_lane_by_id(self, lane_id: Tuple[str, str, int]):
        ...

    def get_env(self):
        ...

    def get_config(self):
        raise NotImplemented
