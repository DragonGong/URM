from .fitting import Fitting
import numpy as np
from typing import List
from .fitting_model_factory import register_fitting
from .fitting import ModelName
from urm.reward.trajectory.traj import TrajEdge, TrajNode

from urm.config import Config
from urm.reward.state.utils.velocity import Velocity


@register_fitting(ModelName.Polynomial)
class Polynomial(Fitting):
    def __init__(self, config: Config.RewardConfig.FittingModelConfigs, **kwargs):
        super().__init__(config, **kwargs)

    def fit_edge_by_node(self, edge: TrajEdge) -> List[TrajNode]:
        """
        code by gpt,test by a man
        使用 Frenet 中的多项式方法拟合 edge 并把离散点写回 edge.discrete_points。
        返回 TrajNode 列表（包含端点）。
        """
        # parameters / safety checks
        p0 = np.array([edge.node_begin.x, edge.node_begin.y], dtype=float)
        pf = np.array([edge.node_end.x, edge.node_end.y], dtype=float)

        # duration T: 优先使用节点时间差，否则 fallback 到 self.interval_duration
        t0 = float(edge.node_begin.get_time() if hasattr(edge.node_begin, "get_time") else 0.0)
        tf = float(edge.node_end.get_time() if hasattr(edge.node_end, "get_time") else (t0 + self.interval_duration))
        T = tf - t0
        if T <= 1e-8:
            # 如果时间差为 0 或负，退回到 interval_duration（至少 1e-3）
            T = max(self.interval_duration, 1e-3)

        # 采样密度：每秒 10 个样本（可按需改）
        num_points = max(1, int(round(T / self.interval_duration)))
        # 我们会生成 num_points+1 个点（包含端点）
        N = num_points

        # 获取速度向量（可能为 None）
        def vel_to_np(v):
            if v is None:
                return np.array([0.0, 0.0], dtype=float)
            if isinstance(v, Velocity):
                return np.array([v.vx, v.vy], dtype=float)
            # 若外部传入 tuple/list
            return np.array(v, dtype=float)

        v0_vec = vel_to_np(getattr(edge.node_begin, "velocity", None))
        vf_vec = vel_to_np(getattr(edge.node_end, "velocity", None))

        # 如果起点速度几乎为 0，则用 p0->pf 的方向作为纵向方向
        diff = pf - p0
        diff_norm = np.linalg.norm(diff)
        v0_norm = np.linalg.norm(v0_vec)
        eps = 1e-8

        if v0_norm > eps:
            e_s = v0_vec / v0_norm
        elif diff_norm > eps:
            e_s = diff / diff_norm
        else:
            # 极端退化，选择 x 轴为纵向
            e_s = np.array([1.0, 0.0], dtype=float)

        # 横向单位向量（右手坐标系，+90deg）
        e_d = np.array([-e_s[1], e_s[0]], dtype=float)

        # Frenet 起终点坐标（以 p0 为原点，因此 s0 = d0 = 0）
        s0 = 0.0
        d0 = 0.0
        rel_pf = pf - p0
        s_f = float(np.dot(rel_pf, e_s))
        d_f = float(np.dot(rel_pf, e_d))

        # 初末速度在 Frenet 分量
        vs0 = float(np.dot(v0_vec, e_s))
        vd0 = float(np.dot(v0_vec, e_d))
        vs_f = float(np.dot(vf_vec, e_s)) if np.linalg.norm(vf_vec) > eps else vs0
        vd_f = float(np.dot(vf_vec, e_d)) if np.linalg.norm(vf_vec) > eps else vd0

        # 假设初末加速度未知，设为 0（如有加速度信息可以替换）
        as0 = 0.0
        as_f = 0.0
        dd0 = 0.0
        ddf = 0.0

        # ----------------------------
        # 横向 d(t) : quintic
        # d(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
        # 6 个边界条件：d(0), d'(0), d''(0), d(T), d'(T), d''(T)
        # ----------------------------
        M = np.zeros((6, 6), dtype=float)
        # t = 0 条件
        M[0, :] = [1, 0, 0, 0, 0, 0]  # d(0)
        M[1, :] = [0, 1, 0, 0, 0, 0]  # d'(0)
        M[2, :] = [0, 0, 2, 0, 0, 0]  # d''(0)
        # t = T 条件
        Tpow = np.array([1, T, T ** 2, T ** 3, T ** 4, T ** 5], dtype=float)
        M[3, :] = Tpow  # d(T)
        M[4, :] = [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4]  # d'(T)
        M[5, :] = [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # d''(T)

        b_vec = np.array([d0, vd0, dd0, d_f, vd_f, ddf], dtype=float)

        try:
            a_coeffs = np.linalg.solve(M, b_vec)
        except np.linalg.LinAlgError:
            # 如果矩阵退化，退化为最小二乘求解
            a_coeffs, *_ = np.linalg.lstsq(M, b_vec, rcond=None)

        # ----------------------------
        # 纵向 s(t) : quartic
        # s(t) = b0 + b1 t + b2 t^2 + b3 t^3 + b4 t^4
        # 边界条件选择：s(0), s'(0), s''(0), s'(T), s''(T)=0（按图片方法）
        # 由于 b0,b1,b2 可由初始条件直接确定，我们用两条方程解 b3,b4
        # ----------------------------
        b0 = s0
        b1 = vs0
        b2 = as0 / 2.0

        # 构造方程:
        # s'(T) = b1 + 2 b2 T + 3 b3 T^2 + 4 b4 T^3 = vs_f
        # s''(T) = 2 b2 + 6 b3 T + 12 b4 T^2 = 0
        A2 = np.array([
            [3 * T ** 2, 4 * T ** 3],
            [6 * T, 12 * T ** 2]
        ], dtype=float)
        rhs2 = np.array([
            vs_f - (b1 + 2 * b2 * T),
            -2 * b2
        ], dtype=float)

        try:
            sol_b34 = np.linalg.solve(A2, rhs2)
            b3, b4 = float(sol_b34[0]), float(sol_b34[1])
        except np.linalg.LinAlgError:
            sol_b34, *_ = np.linalg.lstsq(A2, rhs2, rcond=None)
            b3, b4 = float(sol_b34[0]), float(sol_b34[1])

        b_coeffs = np.array([b0, b1, b2, b3, b4], dtype=float)

        # ----------------------------
        # 采样并转换回 XY，计算速度向量
        # ----------------------------
        nodes: List[TrajNode] = []
        for i in range(N + 1):
            local_t = (i / N) * T  # 0..T
            # evaluate d(t)
            t = local_t
            d_t = (a_coeffs[0] + a_coeffs[1] * t + a_coeffs[2] * t ** 2 +
                   a_coeffs[3] * t ** 3 + a_coeffs[4] * t ** 4 + a_coeffs[5] * t ** 5)
            # d'(t)
            d_dot = (a_coeffs[1] + 2 * a_coeffs[2] * t + 3 * a_coeffs[3] * t ** 2 +
                     4 * a_coeffs[4] * t ** 3 + 5 * a_coeffs[5] * t ** 4)

            # evaluate s(t)
            s_t = (b_coeffs[0] + b_coeffs[1] * t + b_coeffs[2] * t ** 2 +
                   b_coeffs[3] * t ** 3 + b_coeffs[4] * t ** 4)
            # s'(t)
            s_dot = (b_coeffs[1] + 2 * b_coeffs[2] * t + 3 * b_coeffs[3] * t ** 2 + 4 * b_coeffs[4] * t ** 3)

            # convert to XY
            pos_xy = p0 + e_s * s_t + e_d * d_t
            vel_xy = e_s * s_dot + e_d * d_dot

            node = TrajNode(float(pos_xy[0]), float(pos_xy[1]))
            node.set_velocity(Velocity.from_tuple((float(vel_xy[0]), float(vel_xy[1]))))
            # 时间用起始时间加上 local_t（保持 float 精度）
            node.set_timestep(t0 + local_t)
            nodes.append(node)

        # write back to edge
        edge.discrete_points = nodes
        # 保存拟合算法标识，__name__ 在 __repr__ 中会被使用
        edge.fitting_algorithm = self.__class__

        return nodes
