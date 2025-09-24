from enum import Enum, unique

@unique  # 确保枚举值唯一，避免重复定义
class BehaviorName(Enum):
    """行为名称枚举类，定义所有支持的行为类型"""
    LATERAL_KEEP = "lateral_keep"      # 保持当前车道
    LATERAL_LEFT = "lateral_left"      # 向左变道
    LATERAL_RIGHT = "lateral_right"    # 向右变道
    CRUISE = "cruise"                  # 巡航（保持当前速度）
    SOFT_ACCEL = "soft_accel"          # 轻微加速
    HARD_ACCEL = "hard_accel"          # 强烈加速
    SOFT_DECEL = "soft_decel"          # 轻微减速
    HARD_DECEL = "hard_decel"          # 强烈减速

    def __str__(self) -> str:
        """返回枚举对应的字符串值，方便序列化和打印"""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> 'BehaviorName':
        """从字符串创建枚举实例，用于反序列化"""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"无效的行为名称: {value}，有效名称为: {[m.value for m in cls]}")
