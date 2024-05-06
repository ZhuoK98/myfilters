import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.AbstractClass.AbstractFilter import AbstractFilter
import numpy as np


class ParticleFilter(AbstractFilter):

    def __init__(self, X, Q, R, Fx, Hx, NumOfPoint=10, W=0):
        self._NumOfPoint = NumOfPoint
        self._X = np.mat(np.zeros((X.shape[0], NumOfPoint)))
        for i in range(0, NumOfPoint):
            self._X[:, i] = np.mat(X)
        self._x = np.mat(X)
        self._Q = np.mat(Q)
        self._R = np.mat(R)
        self._Fx = Fx
        self._Hx = Hx

    def pf_resample(self, w, x):
        NumOfPoint = x.shape[1]
        c = np.zeros((1, NumOfPoint))
        c[0, 0] = w[0, 0]
        for j in range(1, NumOfPoint):
            c[0, j] = c[0, j - 1] + w[0, j]
        xnew = np.mat(np.zeros(x.shape))
        for j in range(0, NumOfPoint):
            a = np.random.uniform(0, 1)
            for k in range(0, NumOfPoint):
                if a < c[0, k]:
                    xnew[:, j] = x[:, k]
                    break
        return xnew

    def predict(self, i, u=None, t=None):

        for k in range(1, self._NumOfPoint):
            self._X[:, k] = self._Fx(self._X[:, k], u, t, i) + np.random.normal(0, self._Q, self._x.shape)
            # self._X[:, i] = self._A * self._X[:, i] + self._B * u + np.random.normal(0, self._Q, self._x.shape)

    def update(self, i, z, u):

        w = np.mat(np.zeros((1, self._NumOfPoint)))

        for k in range(0, self._NumOfPoint):
            w[:, k] = np.exp(-(z - self._Hx(self._X[:, k], u, i)) ** 2 / (2 * self._R))
        w = w / np.sum(w)

        # 重采样
        result = self.pf_resample(w, self._X)
        self._x = np.sum(result, axis=1) / self._NumOfPoint

        # 更新权重
        # for j in range(0, self._NumOfPoint):
        #     w[0, j] = 1 / self._NumOfPoint

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
        return x

    def Hx(x, u, i):
        return H * x

    kalman_filter = ParticleFilter(X[:, 0], np.mat(np.diag(Q)).T, R, Fx, Hx)

    U = np.tile(U, 70)
    res1 = kalman_filter(Z, U, t)

    end = time.time()
    print("运行耗时", end - start)

    print('滤波之后即预测优化值')
    print(np.array(res1)[0])

    plot_figure(t, np.array(res1)[0], np.array(X)[0], true_value, "PF", "粒子滤波示意图")
