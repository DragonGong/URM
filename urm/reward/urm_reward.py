import numpy as np

# 下面代码全部gpt生成
# ============ Quintic Polynomial ============
class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([
            [T ** 3, T ** 4, T ** 5],
            [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
            [6 * T, 12 * T ** 2, 20 * T ** 3]
        ])
        b = np.array([
            xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
            vxe - self.a1 - 2 * self.a2 * T,
            axe - 2 * self.a2
        ])
        x = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = x

    def calc_point(self, t):
        return (self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 +
                self.a4 * t ** 4 + self.a5 * t ** 5)


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


def URM_reward(ego_state, surrounding_states):
    candidate_trajs = generate_candidate_trajectories(ego_state)
    others_trajs = predict_other_vehicles(surrounding_states)

    risks = [compute_risk(traj, others_trajs) for traj in candidate_trajs]
    URM_value = np.mean(risks)
    reward = -URM_value
    return reward
