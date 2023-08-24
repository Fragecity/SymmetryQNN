import numpy as np
from TrainUtils.Optmizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, useGPU=False):
        super().__init__(useGPU=useGPU)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def val_shift(self, gradient):
        if self.m is None or self.v is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        m_corrected = self.m / (1 - self.beta1 ** self.t)
        v_corrected = self.v / (1 - self.beta2 ** self.t)

        return -self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
