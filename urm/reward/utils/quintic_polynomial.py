import numpy as np


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
