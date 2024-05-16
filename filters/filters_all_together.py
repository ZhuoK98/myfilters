import numpy as np
from abc import ABC, abstractmethod


class AbstractFilter(ABC):

    @abstractmethod
    def __init__(self, X, P, Q, R, Fx, Hx):
        self._x = np.mat(X)
        self._P = np.mat(P)
        self._Q = np.mat(Q)
        self._R = np.mat(R)
        self._Fx = Fx
        self._Hx = Hx

    def __call__(self, z_array, u_array=None, t_array=None):
        z_array = np.mat(z_array)
        u_array = np.mat(u_array)
        z_arr_len = z_array.shape[1]
        res = np.mat(np.zeros((self.x.shape[0], z_arr_len)))

        for i in range(0, z_arr_len):
            X = self.step(i, z_array[:, i], u_array[:, i], t_array[i])
            res[:, i] = X
        return res

    def step(self, i, z, u=None, t=None):

        self.predict(i, u, t)
        self.update(i, z, u)
        return self.x

    @abstractmethod
    def update(self, i, z, u):
        pass

    @abstractmethod
    def predict(self, i, u=None, t=None):
        pass

    @property
    def x(self):
        return self._x


class AbstractExtendedFilter(AbstractFilter):

    def __init__(self, X, P, Q, R, Fx, Hx, HJacobian):
        super().__init__(X, P, Q, R, Fx, Hx)
        self._HJacobian = HJacobian

    @abstractmethod
    def update(self, i, z, u):
        pass

    @abstractmethod
    def predict(self, i, u=None, t=None):
        pass


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


class ExtendedKalmanFilter(KalmanFilter, AbstractExtendedFilter):
    __init__ = AbstractExtendedFilter.__init__


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
    def update(self, i, z, u):

        # S = np.linalg.cholesky(self._P)
        # X = self._x + S * self._kexi

        # Zhat = self._Hx(X, u, i)
        Zhat = self._Hx(self._Xhat, u, i)
        zhat = np.average(Zhat, axis=1)

        S = self.self_variance(Zhat, zhat) + self._R
        # Pxz = self.cross_variance(X, Zhat, self._x, zhat)
        Pxz = self.cross_variance(self._Xhat, Zhat, self._x, zhat)
        K = Pxz * S.I

        self._x = self._x + K * (z - zhat)
        self._P = self._P - K * S * K.T

    # 此实现参考标准CKF
    # def update(self, i, z, u):
    #
    #     S = np.linalg.cholesky(self._P)
    #     X = self._x + S * self._kexi
    #
    #     Zhat = self._Hx(X, u, i)
    #     zhat = np.average(Zhat, axis=1)
    #
    #     S = self.self_variance(Zhat, zhat) + self._R
    #     Pxz = self.cross_variance(X, Zhat, self._x, zhat)
    #     K = Pxz * S.I
    #
    #     self._x = self._x + K * (z - zhat)
    #     self._P = self._P - K * S * K.T

    def predict(self, i, u=None, t=None):

        S = np.linalg.cholesky(self._P)
        X = self._x + S * self._kexi

        Xhat = self._Xhat = self._Fx(X, u, t, i)

        self._x = np.average(Xhat, axis=1)
        self._P = self.self_variance(Xhat, self._x) + self._Q


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


class ExtendedSlidingInnovationFilter(SlidingInnovationFilter, AbstractExtendedFilter):

    def __init__(self, X, P, Q, R, Fx, delta, Hx, HJacobian):
        AbstractExtendedFilter.__init__(self, X, P, Q, R, Fx, Hx, HJacobian)
        self._delta = np.mat(delta)


class AdaptiveExtendedSlidingInnovationFilter(AdaptiveSlidingInnovationFilter, AbstractExtendedFilter):
    __init__ = AbstractExtendedFilter.__init__



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
        self._x = np.sum(result, axis=1)/self._NumOfPoint

        # 更新权重
        # for j in range(0, self._NumOfPoint):
        #     w[0, j] = 1 / self._NumOfPoint

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    start = time.time()

    np.random.seed(42)  # 固定随机种子，确保每次生成的随机数据一样
    delta_t = 0.1  # 每秒采样一次
    end_t = 7  # 时间长度
    time_t = end_t * 10  # 采样次数
    t = np.arange(0, end_t, delta_t)  # 设置时间长度
    v_var = 4  # 测量噪声的方差
    v_noise = np.round(np.random.normal(0, v_var, time_t), 2)  # 定义测量噪声
    a = 1  # 加速度
    s = np.add((1 / 2 * a * t ** 2), v_noise)  # 定义测量的位置
    v = a * t  # 定义速度数组

    X = np.mat([s, v])  # 定义状态矩阵
    U = np.mat([[a], [a]])  # 定义外界对系统作用矩阵
    A = np.mat([[1, delta_t], [0, 1]])  # 定义状态转移矩阵
    B = np.mat([[1 / 2 * (delta_t ** 2), 0], [0, delta_t]])  # 定义输入控制矩阵
    P = np.mat([[1, 0], [0, 1]])  # 定义初始协方差矩阵
    Q = np.mat([[0.1, 0], [0, 0.1]])  # 定义测量噪声协方差矩阵
    H = np.mat([1.0, 0])  # 定义观测矩阵
    R = np.mat([1])  # 定义观测噪声协方差矩阵
    Z = H * X

    print('滤波之前即测量值')
    print(np.array(X)[0])


    def Fx(x, u, t, i):
        x = A * x + B * u
        return A, x

    def Fx1(x, u, t, i):
        _1, _2 = Fx(x, u, t, i)
        return _2

    def Hx(x, u, i):
        return H * x


    kf = KalmanFilter(X[:, 0], P, Q, R, H, Fx, Hx)
    ekf = ExtendedKalmanFilter(X[:, 0], P, Q, R, Fx, Hx, lambda x: H)

    ckf = CubatureKalmanFilter(X[:, 0], P, Q, R, Fx1, Hx)
    ukf = UnscentedKalmanFilter(X[:, 0], P, Q, R, Fx1, Hx)
    pf = ParticleFilter(X[:, 0], np.mat(np.diag(Q)).T, R, Fx1, Hx, 10)

    sif = SlidingInnovationFilter(X[:, 0], P, Q, R, H, Fx, Hx, 200)
    esif = ExtendedSlidingInnovationFilter(X[:, 0], P, Q, R, Fx, 200, Hx, lambda x: H)
    asif = AdaptiveSlidingInnovationFilter(X[:, 0], P, Q, R, H, Fx, Hx)
    aesif = AdaptiveExtendedSlidingInnovationFilter(X[:, 0], P, Q, R, Fx, Hx, lambda x: H)

    U = np.tile(U, 70)

    algorithms = {
                  # 'KF': kf,
                  'EKF': ekf,
                  'CKF': ckf,
                  'UKF': ukf,
                  'PF@point=10': pf,
                  # r'SIF@$\delta=200$': sif,
                  r'ESIF@$\delta=200$': esif,
                  # 'ASIF': asif,
                  'AESIF': aesif
                  }

    result = {'true': 1/2*a*t**2,
              'measure': np.array(X)[0]
              }

    for name, algorithm in algorithms.items():
        res = algorithm(Z, U, t)
        result[name] = np.array(res)[0]

    end = time.time()
    print("运行耗时", end - start)

    # print('滤波之后即预测优化值')
    # print(np.array(res2)[0])

    plt.rcParams['font.sans-serif'] = ['SimHei']           	# 设置正常显示中文
    plt.rcParams['axes.unicode_minus'] = False              # 正常显示坐标轴负号
    for label, res in result.items():
        plt.plot(t, res, label=label)

    plt.xlabel("时间")                                     	# 设置X轴的名字
    plt.ylabel("位移")                                     	# 设置Y轴的名字
    plt.title("滤波示意图") 								    # 设置标题
    plt.legend(ncol=3)                                      # 设置图例
    plt.show()                                             	# 显示图表

