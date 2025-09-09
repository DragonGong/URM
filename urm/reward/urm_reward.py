import numpy as np
from urm.reward.utils.quintic_polynomial import QuinticPolynomial


# ============ 轨迹生成器 ============
def generate_candidate_trajectories(ego_state, T=3.0, dt=0.2):
    xs, ys, vxs, vys = ego_state
    candidate_trajs = []

    target_offsets = [0, 4, -4]  # 保持车道、左移、右移
    for d in target_offsets:
        lat_poly = QuinticPolynomial(ys, vys, 0.0, ys + d, 0.0, 0.0, T)
        lon_poly = QuinticPolynomial(xs, vxs, 0.0, xs + vxs * T, vxs, 0.0, T)

        traj = []
        for t in np.arange(0, T + dt, dt):
            x = lon_poly.calc_point(t)
            y = lat_poly.calc_point(t)
            traj.append((x, y, t))
        candidate_trajs.append(traj)

    return candidate_trajs


# ============ 其他车辆轨迹预测 ============
def predict_other_vehicles(surrounding_states, T=3.0, dt=0.2):
    predictions = []
    for s in surrounding_states:
        xs, ys, vxs, vys = s
        traj = []
        for t in np.arange(0, T + dt, dt):
            x = xs + vxs * t
            y = ys + vys * t
            traj.append((x, y, t))
        predictions.append(traj)
    return predictions


# ============ 风险计算 ============
def compute_risk(ego_traj, others_trajs, collision_radius=2.0):
    min_dist = float("inf")
    for (x, y, t) in ego_traj:
        for traj in others_trajs:
            ox, oy, _ = traj[int(t / 0.2)]
            dist = np.linalg.norm([x - ox, y - oy])
            if dist < min_dist:
                min_dist = dist

    if min_dist < collision_radius:
        return 1.0
    elif min_dist < 5.0:
        return 0.7
    elif min_dist < 10.0:
        return 0.3
    else:
        return 0.0


def URM_reward(ego_state, surrounding_states, train_config):
    candidate_trajs = generate_candidate_trajectories(ego_state)
    others_trajs = predict_other_vehicles(surrounding_states)

    risks = [compute_risk(traj, others_trajs) for traj in candidate_trajs]
    URM_value = np.mean(risks)
    R_safe = -URM_value

    # 速度奖励
    ego_speed = np.linalg.norm([ego_state[2], ego_state[3]])
    desired_speed = train_config['desired_speed']  # 目标速度
    R_speed = 1.0 - abs(ego_speed - desired_speed) / desired_speed

    # 变道惩罚
    lateral_velocity = abs(ego_state[3])
    R_lateral = -train_config['v2r_w'] * lateral_velocity
    # 组合奖励
    reward = train_config['r_safe_w'] * R_safe + train_config['r_speed_w'] * R_speed + train_config[
        'r_lateral_w'] * R_lateral
    return reward
