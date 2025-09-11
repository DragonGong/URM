from typing import Protocol, Any, Tuple, Union
from urm.reward.state.utils.position import Position
import numpy as np


class EnvInterface(Protocol):
    def _extract_xy(self, pos: Any) -> Tuple[float, float]:
        ...

    def judge_match_road(
            self,
            pos: Union[Tuple[float, float], Position, np.ndarray, dict],
            margin: float = 0.5
    ) -> bool:
        ...
