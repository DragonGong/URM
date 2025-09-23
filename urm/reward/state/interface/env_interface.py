from typing import Protocol, Any, Tuple, Union, runtime_checkable
import numpy as np


@runtime_checkable
class EnvInterface(Protocol):
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

    def get_frenet_velocity(self, x: float, y: float, vx: float, vy: float) -> Tuple[float, float]:
        ...

    def frenet_velocity_to_cartesian(
            self,
            x: float,
            y: float,
            v_lon: float,
            v_lat: float
    ) -> tuple[np.ndarray[tuple[int, ...], Any], np.ndarray[tuple[int, ...], Any]]:
        ...