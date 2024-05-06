import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.KalmanFilter import KalmanFilter
import numpy as np

class SlidingInnovationFilter(KalmanFilter):

    def __init__(self, X, P, Q, R, H, Fx, Hx, delta):
        super().__init__(X, P, Q, R, H, Fx, Hx)
        self._delta = np.mat(delta)

    def saturateItem(self, a):
        if a > 1:
            a = 1
        elif a < -1:
            a = -1
        return a

    def matrixDiagOperate(self, M, *args, **kwargs):
        m, n = M.shape
        if m != n:
            raise Exception("矩阵不为方阵")
        else:
            res = np.mat(np.zeros((m, n)))
            for i in range(m):
                for j in range(n):
                    if i == j:
                        res[i][j] = kwargs['OperateFunc'](M, i, j, args)

            return res

    def matrixDiagSaturate(self, M, D):
        operatefunc = lambda M, i, j, D: self.saturateItem(abs(M[i][j]) / D[i][j])

        return self.matrixDiagOperate(M, D, OperateFunc=operatefunc)

    def update(self, i, z, u):

        P = self._P
        R = self._R
        x = self._x

        H = self._HJacobian(x)

        hx = self._Hx(x, u, i)
        residual = z - hx
        delta = self.delta(residual)
        K = np.linalg.pinv(H) * self.matrixDiagSaturate(residual, delta)

        self._K = K
        self._x = x + K * residual

        KH = K * H
        I_KH = np.identity((KH).shape[1]) - KH
        self._P = I_KH * P * I_KH.T + K * R * K.T

    def delta(self, residual=None):
        return self._delta

if __name__ == '__main__':
    from filters.test.GenerateData import generate_data
    from filters.test.PlotFigure import plot_figure
    import time

    start = time.time()

    X, U, A, B, P, Q, H, R, Z, t, true_value = generate_data()

    print('滤波之前即测量值')
    print(np.array(X)[0])


    def Fx(x, u, t, i):
        x = A * x + B * u
        return A, x

    def Hx(x, u, i):
        return H * x


    kalman_filter = SlidingInnovationFilter(X[:, 0], P, Q, R, H, Fx, Hx, 200)

    U = np.tile(U, 70)
    res1 = kalman_filter(Z, U, t)

    end = time.time()
    print("运行耗时", end - start)

    print('滤波之后即预测优化值')
    print(np.array(res1)[0])

    plot_figure(t, np.array(res1)[0], np.array(X)[0], true_value, "SIF", "滑动新息滤波示意图")
