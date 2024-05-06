import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.SlidingInnovationFilter import SlidingInnovationFilter
from filters.KalmanFilter import KalmanFilter


class AdaptiveSlidingInnovationFilter(SlidingInnovationFilter):
    __init__ = KalmanFilter.__init__

    def matrixDiagAbs(self, M):
        operatefunc = lambda M, i, j, D: abs(M[i][j])

        return self.matrixDiagOperate(M, OperateFunc=operatefunc)

    def delta(self, residual):
        P = self._P
        R = self._R
        x = self._x
        H = self._HJacobian(x)

        S = H * P * H.T + R

        self._delta = S * (S - R).I * self.matrixDiagAbs(residual)

        return self._delta

if __name__ == '__main__':
    from filters.test.GenerateData import generate_data
    from filters.test.PlotFigure import plot_figure
    import numpy as np
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

    kalman_filter = AdaptiveSlidingInnovationFilter(X[:, 0], P, Q, R, H, Fx, Hx)

    U = np.tile(U, 70)
    res1 = kalman_filter(Z, U, t)


    end = time.time()
    print("运行耗时", end - start)

    print('滤波之后即预测优化值')
    print(np.array(res1)[0])

    plot_figure(t, np.array(res1)[0], np.array(X)[0], true_value, "ASIF", "自适应滑动新息滤波示意图")
