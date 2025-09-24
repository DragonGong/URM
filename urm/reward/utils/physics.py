def calculate_displacement(initial_velocity, final_velocity, time):
    """
    计算匀加速直线运动的位移

    参数:
    initial_velocity (float): 初始速度
    final_velocity (float): 最终速度
    time (float): 运动时间，必须为正数

    返回:
    float: 位移大小

    异常:
    ValueError: 当时间为非正数时抛出
    """
    if time <= 0:
        raise ValueError("时间必须是正数")

    # 匀加速直线运动位移公式：s = (v0 + v1) * t / 2
    displacement = (initial_velocity + final_velocity) * time / 2
    return displacement
