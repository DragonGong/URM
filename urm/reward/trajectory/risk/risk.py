import math
from typing import Dict, Optional, List, Tuple
import bisect


class Risk:
    def __init__(self):
        # 使用字典：时间 t -> (risk_value, speed_at_t)
        self._data: Dict[float, Tuple[float, float]] = {}
        # 可选：保持时间排序的键列表，用于插值或遍历
        self._sorted_times: List[float] = []

    def set_value(self, t: float, risk: float, speed: float):
        """
        设置在时间 t 的风险值和速度
        """
        if t not in self._data:
            bisect.insort(self._sorted_times, t)  # 保持时间有序（可选）
        self._data[t] = (risk, speed)

    def get_value(self, t: float) -> Optional[float]:
        """
        获取时间 t 的风险值，不存在则返回 None
        """
        entry = self._data.get(t)
        return entry[0] if entry is not None else None

    def get_speed(self, t: float) -> Optional[float]:
        """
        获取时间 t 的速度，不存在则返回 None
        """
        entry = self._data.get(t)
        return entry[1] if entry is not None else None

    def get_risk_speed(self, t: float) -> Optional[Tuple[float, float]]:
        """
        同时获取 (risk, speed)
        """
        return self._data.get(t)

    def times(self) -> List[float]:
        """
        返回所有记录的时间点（排序后）
        """
        return self._sorted_times.copy()  # 返回副本防止外部修改

    def items(self) -> List[Tuple[float, float, float]]:
        """
        返回 [(t, risk, speed), ...] 列表，按时间排序
        """
        return [(t, risk, speed) for t in self._sorted_times
                for risk, speed in [self._data[t]]]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
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
            print("matplotlib 未安装，无法绘图")
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
