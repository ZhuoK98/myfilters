import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(PARENT_DIR))

from filters import KalmanFilter, ExtendedKalmanFilter, CubatureKalmanFilter, UnscentedKalmanFilter, \
    SlidingInnovationFilter, ParticleFilter, ExtendedSlidingInnovationFilter, AdaptiveSlidingInnovationFilter, \
    AdaptiveExtendedSlidingInnovationFilter
from filters.test.GenerateData import generate_data
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    X, U, A, B, P, Q, H, R, Z, t, true_value = generate_data()

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

    result = {'true': true_value,
              'measure': np.array(X)[0]
              }

    for name, algorithm in algorithms.items():
        res = algorithm(Z, U, t)
        result[name] = np.array(res)[0]

    for label, res in result.items():
        plt.plot(t, res, label=label)

    plt.legend(ncol=3)
    plt.show()
