import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.AbstractClass.AbstractFilter import AbstractFilter
import numpy as np

class UnscentedKalmanFilter(AbstractFilter):

    def __init__(self, X, P, Q, R, Fx, Hx, la=None, alpha=1, beta=0, kappa=1):
        super().__init__(X, P, Q, R, Fx, Hx)

        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        n = X.shape[0]
        if la is None:
            self._lambda = self._alpha * self._alpha * (n + self._kappa) - n
        else:
            self._lambda = la

        self._Wm, self._Wc = self.compute_weight(self._lambda, n, self._alpha, self._beta)

    def compute_weight(self, la, n, alpha, beta):
        wm = [0] * (2*n+1)
        wc = [0] * (2*n+1)

        wm[0] = la / (n + la)
        wc[0] = la / (n + la) + 1 - alpha * alpha + beta

        for i in range(1, 2*n+1):
            wm[i] = wc[i] = 1 / (2 * (n + la))

        return wm, wc

    def unscented_transform(self, sigmas, Wm, Wc, noise_cov=None):
        m, n = sigmas.shape

        x = np.mat(np.zeros((m, 1)))

        for i in range(0, n):
            x += Wm[i] * sigmas[:, i]

        P = self.self_variance(sigmas, x, Wc)

        if noise_cov is not None:
            P += noise_cov

        return x, P

    def self_variance(self, x, mu_x, Wc):
        m, n = x.shape

        P = np.mat(np.zeros((m, m)))

        for i in range(0, n):
            residual_x = x[:, i] - mu_x
            P += Wc[i] * residual_x * residual_x.T

        return P

    def cross_variance(self, x, z, mu_x, mu_z, Wc):

        n = x.shape[0]
        m = z.shape[0]

        Pxz = np.mat(np.zeros((n, m)))

        for i in range(0, x.shape[1]):
            residual_x = x[:, i] - mu_x
            residual_z = z[:, i] - mu_z
            Pxz += Wc[i] * residual_x * residual_z.T

        return Pxz

    def compute_sigmas(self, x, p, f, la):
        n = x.shape[0]
        m = f(x).shape[0]

        sigmas_tmp = np.mat(np.zeros((n, 2*n+1)))
        sigmas = np.mat(np.zeros((m, 2*n+1)))

        sigmas_tmp[:, 0] = x

        lower = np.linalg.cholesky(p)
        gamma = np.sqrt((n + la))

        for i in range(1, n+1):
            sigmas_tmp[:, i] = x + gamma * lower[:, i - 1]
            sigmas_tmp[:, i + n] = x - gamma * lower[:, i - 1]

        self._sigmas_x = sigmas_tmp

        for i in range(0, sigmas.shape[1]):
            sigmas[:, i] = f(sigmas_tmp[:, i])

        return sigmas

    def update(self, i, z, u):

        sigmas_z = self.compute_sigmas(self._x, self._P, lambda x: self._Hx(x, u, i), self._lambda)
        mu_z, Pz = self.unscented_transform(sigmas_z, self._Wm, self._Wc, self._R)

        Pxz = self.cross_variance(self._sigmas_x, sigmas_z, self._x, mu_z, self._Wc)
        K = Pxz * Pz.I

        self._x = self._x + K * (z - mu_z)
        self._P = self._P - K * Pz * K.T

    def predict(self, i, u=None, t=None):

        self._sigmas_x = self.compute_sigmas(self._x, self._P, lambda x: self._Fx(x, u, t, i), self._lambda)
        self._x, self._P = self.unscented_transform(self._sigmas_x, self._Wm, self._Wc, self._Q)

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


    kalman_filter = UnscentedKalmanFilter(X[:, 0], P, Q, R, Fx, Hx)

    U = np.tile(U, 70)
    res1 = kalman_filter(Z, U, t)

    end = time.time()
    print("运行耗时", end - start)

    print('滤波之后即预测优化值')

    print(np.array(res1)[0])

    plot_figure(t, np.array(res1)[0], np.array(X)[0], true_value, "UKF", "无迹卡尔曼滤波示意图")
