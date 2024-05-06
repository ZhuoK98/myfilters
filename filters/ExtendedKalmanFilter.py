import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.AbstractClass.AbstractExtendedFilter import AbstractExtendedFilter
from filters.KalmanFilter import KalmanFilter


class ExtendedKalmanFilter(KalmanFilter, AbstractExtendedFilter):
    __init__ = AbstractExtendedFilter.__init__


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

    kalman_filter1 = ExtendedKalmanFilter(X[:, 0], P, Q, R, Fx, Hx, lambda x: H)

    U = np.tile(U, 70)
    res1 = kalman_filter1(Z, U, t)

    end = time.time()
    print("运行耗时", end - start)

    print('滤波之后即预测优化值')
    print(np.array(res1)[0])

    plot_figure(t, np.array(res1)[0], np.array(X)[0], true_value, "EKF", "扩展卡尔曼滤波示意图")
