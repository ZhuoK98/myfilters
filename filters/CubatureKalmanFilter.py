import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.AbstractClass.AbstractFilter import AbstractFilter
import numpy as np

class CubatureKalmanFilter(AbstractFilter):

    def __init__(self, X, P, Q, R, Fx, Hx):
        super().__init__(X, P, Q, R, Fx, Hx)
        dimemsion = X.shape[0]
        self._kexi = np.mat(np.sqrt(dimemsion) * np.append(np.eye(dimemsion), -np.eye(dimemsion), axis=1))    # 采样点

    def self_variance(self, x, mu_x, Wc=None):
        m, n = x.shape

        if Wc is None:
            Wc = [1/x.shape[1] for i in range(0, x.shape[1])]

        P = np.mat(np.zeros((m, m)))

        for i in range(0, n):
            residual_x = x[:, i] - mu_x
            P += Wc[i] * residual_x * residual_x.T

        return P

    def cross_variance(self, x, z, mu_x, mu_z, Wc=None):

        n = x.shape[0]
        m = z.shape[0]

        if Wc is None:
            Wc = [1/x.shape[1] for i in range(0, x.shape[1])]

        Pxz = np.mat(np.zeros((n, m)))

        for i in range(0, x.shape[1]):
            residual_x = x[:, i] - mu_x
            residual_z = z[:, i] - mu_z
            Pxz += Wc[i] * residual_x * residual_z.T

        return Pxz

    # 此实现参考 http://github.com/rlabbe/filterpy 与标准CKF略有不同？
    # def update(self, i, z, u):
    #
    #     # S = np.linalg.cholesky(self._P)
    #     # X = self._x + S * self._kexi
    #
    #     # Zhat = self._Hx(X, u, i)
    #     Zhat = self._Hx(self._Xhat, u, i)
    #     zhat = np.average(Zhat, axis=1)
    #
    #     S = self.self_variance(Zhat, zhat) + self._R
    #     # Pxz = self.cross_variance(X, Zhat, self._x, zhat)
    #     Pxz = self.cross_variance(self._Xhat, Zhat, self._x, zhat)
    #     K = Pxz * S.I
    #
    #     self._x = self._x + K * (z - zhat)
    #     self._P = self._P - K * S * K.T

    # 此实现参考标准CKF
    def update(self, i, z, u):

        S = np.linalg.cholesky(self._P)
        X = self._x + S * self._kexi

        Zhat = self._Hx(X, u, i)
        zhat = np.average(Zhat, axis=1)

        S = self.self_variance(Zhat, zhat) + self._R
        Pxz = self.cross_variance(X, Zhat, self._x, zhat)
        K = Pxz * S.I

        self._x = self._x + K * (z - zhat)
        self._P = self._P - K * S * K.T

    def predict(self, i, u=None, t=None):

        S = np.linalg.cholesky(self._P)
        X = self._x + S * self._kexi

        Xhat = self._Xhat = self._Fx(X, u, t, i)

        self._x = np.average(Xhat, axis=1)
        self._P = self.self_variance(Xhat, self._x) + self._Q

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


    kalman_filter = CubatureKalmanFilter(X[:, 0], P, Q, R, Fx, Hx)

    U = np.tile(U, 70)
    res1 = kalman_filter(Z, U, t)

    end = time.time()
    print("运行耗时", end - start)

    print('滤波之后即预测优化值')
    print(np.array(res1)[0])

    plot_figure(t, np.array(res1)[0], np.array(X)[0], true_value, "CKF", "容积卡尔曼滤波示意图")
