import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.AbstractClass.AbstractFilter import AbstractFilter
import numpy as np


class KalmanFilter(AbstractFilter):

    def __init__(self, X, P, Q, R, H, Fx, Hx):
        super().__init__(X, P, Q, R, Fx, Hx)
        self._H = H
        self._HJacobian = lambda x: H

    def update(self, i, z, u):

        P = self._P
        R = self._R
        x = self._x

        H = self._HJacobian(x)

        S = H * P * H.T + R
        K = P * H.T * S.I
        self._K = K

        hx = self._Hx(x, u, i)
        residual = np.subtract(z, hx)
        self._x = x + K * residual

        KH = K * H
        I_KH = np.identity((KH).shape[1]) - KH
        self._P = I_KH * P * I_KH.T + K * R * K.T

    def predict(self, i, u=None, t=None):

        self._A, self._x = self._Fx(self._x, u, t, i)
        self._P = self._A * self._P * self._A.T + self._Q

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

    kalman_filter3 = KalmanFilter(X[:, 0], P, Q, R, H, Fx, Hx)
    U = np.tile(U, 70)

    res3 = kalman_filter3(Z, U, t)

    end = time.time()
    print("运行耗时", end - start)

    print('滤波之后即预测优化值')
    print(np.array(res3)[0])

    plot_figure(t, np.array(res3)[0], np.array(X)[0], true_value, "KF", "卡尔曼滤波示意图")
