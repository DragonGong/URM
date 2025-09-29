import math
from typing import Dict, Optional, List, Tuple
import bisect


class Risk:
    def __init__(self, single_mode: bool = True):
        """
        :param single_mode: 是否启用单点模式（只存储一个时间点的数据）
        """
        self._single_mode = single_mode

        if single_mode:
            # 单点模式：只存一个时间点
            self._time: Optional[float] = None
            self._risk_value: Optional[float] = None
            self._speed: Optional[float] = None
        else:
            # 多点模式：兼容原逻辑
            self._data: Dict[float, Tuple[float, float]] = {}
            self._sorted_times: List[float] = []

    def set_value(self, t: float, risk: float, speed: float):
        """
        设置在时间 t 的风险值和速度
        """
        if self._single_mode:
            # 单点模式：直接覆盖
            self._time = t
            self._risk_value = risk
            self._speed = speed
        else:
            # 多点模式：原逻辑
            if t not in self._data:
                bisect.insort(self._sorted_times, t)
            self._data[t] = (risk, speed)

    def get_value(self, t: Optional[float] = None) -> Optional[float]:
        """
        获取风险值
        - 单点模式：忽略 t，直接返回当前值
        - 多点模式：需指定 t
        """
        if self._single_mode:
            return self._risk_value
        else:
            if t is None:
                raise ValueError("多点模式下必须指定时间 t")
            entry = self._data.get(t)
            return entry[0] if entry is not None else None

    def get_speed(self, t: Optional[float] = None) -> Optional[float]:
        """
        获取速度
        - 单点模式：忽略 t
        - 多点模式：需指定 t
        """
        if self._single_mode:
            return self._speed
        else:
            if t is None:
                raise ValueError("多点模式下必须指定时间 t")
            entry = self._data.get(t)
            return entry[1] if entry is not None else None

    def get_risk_speed(self, t: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """
        同时获取 (risk, speed)
        """
        if self._single_mode:
            if self._time is None:
                return None
            return (self._risk_value, self._speed)
        else:
            if t is None:
                raise ValueError("多点模式下必须指定时间 t")
            return self._data.get(t)

    def times(self) -> List[float]:
        """
        返回所有记录的时间点（排序后）
        """
        if self._single_mode:
            return [self._time] if self._time is not None else []
        else:
            return self._sorted_times.copy()

    def items(self) -> List[Tuple[float, float, float]]:
        """
        返回 [(t, risk, speed), ...] 列表，按时间排序
        """
        if self._single_mode:
            if self._time is None:
                return []
            return [(self._time, self._risk_value, self._speed)]
        else:
            return [(t, risk, speed) for t in self._sorted_times
                    for risk, speed in [self._data[t]]]

    def __len__(self) -> int:
        if self._single_mode:
            return 1 if self._time is not None else 0
        else:
            return len(self._data)

    def __repr__(self) -> str:
        if self._single_mode:
            if self._time is None:
                return "Risk(single_mode=True, empty)"
            return f"Risk(single_mode=True, t={self._time:.2f}, risk={self._risk_value:.3f}, speed={self._speed:.2f})"
        else:
            if not self._data:
                return "Risk()"
            items_str = ", ".join(f"t={t:.2f}: risk={r:.3f}, speed={s:.2f}"
                                  for t, (r, s) in sorted(self._data.items()))
            return f"Risk({items_str})"

    def interpolate(self, t: float) -> Optional[Tuple[float, float]]:
        """
        线性插值获取任意时间点的风险和速度（如果支持）
        要求至少有两个点
        """
        assert self._single_mode is not True, "this risk is single mode"
        if len(self._sorted_times) < 2:
            return None

        idx = bisect.bisect(self._sorted_times, t)
        if idx == 0:
            t0 = self._sorted_times[0]
            return self._data[t0]
        if idx >= len(self._sorted_times):
            t_last = self._sorted_times[-1]
            return self._data[t_last]

        t1 = self._sorted_times[idx - 1]
        t2 = self._sorted_times[idx]
        r1, s1 = self._data[t1]
        r2, s2 = self._data[t2]

        # 线性插值
        ratio = (t - t1) / (t2 - t1)
        risk_interp = r1 + ratio * (r2 - r1)
        speed_interp = s1 + ratio * (s2 - s1)

        return risk_interp, speed_interp

    def plot(self, show_speed=True, show_risk=True):
        """
        绘制 risk(t) 和 speed(t) 曲线（需 matplotlib）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.error("matplotlib 未安装，无法绘图")
            return

        times = self.times()
        risks = [self.get_value(t) for t in times]
        speeds = [self.get_speed(t) for t in times]

        plt.figure(figsize=(10, 4))
        if show_risk:
            plt.plot(times, risks, label='Risk', marker='o', color='red')
        if show_speed:
            plt.plot(times, speeds, label='Speed', marker='s', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('Risk and Speed over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def calculate_last_risk_value(current_risk: float, v_t: float,
                                  velocity_unit: float, discount_factor_max: float, discount_factor_min: float):
        return current_risk * max(discount_factor_max - v_t / velocity_unit, discount_factor_min)
