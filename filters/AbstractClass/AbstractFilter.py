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