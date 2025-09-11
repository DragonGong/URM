from typing import Protocol, Any, Tuple, Union, runtime_checkable
from urm.reward.state.utils.position import Position
import numpy as np


@runtime_checkable
class EnvInterface(Protocol):
    def _extract_xy(self, pos: Any) -> Tuple[float, float]:
        ...

    def judge_match_road(
            self,
            pos: Union[Tuple[float, float], Position, np.ndarray, dict],
            margin: float = 0.5
    ) -> bool:
        ...
