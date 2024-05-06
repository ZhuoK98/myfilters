import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from AbstractClass.AbstractFilter import AbstractFilter
from abc import abstractmethod


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
