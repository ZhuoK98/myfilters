import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from filters.AdaptiveExtendedSlidingInnovationFilter import AdaptiveExtendedSlidingInnovationFilter
from filters.AbstractClass.AbstractExtendedFilter import AbstractExtendedFilter
import numpy as np
class AmendedAdaptiveExtendedSlidingInnovationFilter(AdaptiveExtendedSlidingInnovationFilter):

    def __init__(self, X, P, Q, R, Fx, Hx, HJacobian):
        AbstractExtendedFilter.__init__(self, X, P, Q, R, Fx, Hx, HJacobian)
        self._last_soc = X[2, 0]
        self._last_u = 0
        self._PID_error = [0] * 3
        self._PID_Kp = 20
        self._PID_Ki = 0.01
        self._PID_Kd = 0

    def update(self, i, z, u):

        P = self._P
        R = self._R
        x = self._x
        A = self._A

        H = self._HJacobian(x)
        hx = self._Hx(x, u, i)
        residual = z - hx

        self._PID_error[0] = self._PID_error[1]
        self._PID_error[1] = self._PID_error[2]
        self._PID_error[2] = self._last_soc - x[2, 0]
        PID_u = self._last_u + self._PID_Kp * (self._PID_error[2] - self._PID_error[1]) + self._PID_Ki * self._PID_error[2] + \
                self._PID_Kd * (self._PID_error[2] - 2 * self._PID_error[1] + self._PID_error[0])
        delta = self.delta(residual) * abs(1 + PID_u)

        self._last_u = PID_u
        self._last_soc = x[2, 0]

        K = np.linalg.pinv(H) * self.matrixDiagSaturate(residual, delta)

        # self.delta_last = delta

        self._K = K
        self._x = x + K * residual

        KH = K * H
        I_KH = np.identity((KH).shape[1]) - KH
        self._P = I_KH * P * I_KH.T + K * R * K.T

        alpha = 0.8
        # self._R = alpha * R + (1 - alpha) * (residual * residual.T - H * P * H.T)
        res = z - self._Hx(self._x, u, i)
        self._R = alpha * R + (1-alpha)*(res*res.T + H * self._P * H.T)

        # alpha = 0.9999
        # self._Q = alpha * self._Q + (1-alpha)*(K * res*res.T * K.T)

        epsilon = 0.0121 / 3600
        self._P[2, 2] = self._P[2, 2] - epsilon

        # Amend
        # S = H * P * H.T + R
        # W = np.linalg.pinv(H*A) * S.I * H * P * H.T
        # self._x = self._x + W * residual
        # # # AW = A * W
        # # # P = self._P = P + AW * S * AW.T - P * H.T * AW.T - AW * H * P
        # # # P = P_tmp
        # # recalculate
        # x = self._x
        # hx = self._Hx(x, u, i)
        # residual = z - hx
        # delta_tmp = self.delta(residual)
        # if delta_tmp > delta:
        #     delta = delta_tmp
        #
        # K = np.linalg.pinv(H) * self.matrixDiagSaturate(residual, delta)
        # self._K = K
        # self._x = x + K * residual
        #
        # KH = K * H
        # I_KH = np.identity((KH).shape[1]) - KH
        # self._P = I_KH * P * I_KH.T + K * R * K.T