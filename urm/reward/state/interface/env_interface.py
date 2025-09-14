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
