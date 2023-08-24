import numpy as np
# import cupy as cp
from multiprocessing import Pool
import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     import cupy as cp



# mp.get_context().Process().reducer = dill

class Optimizer:
    """Base class for optimizers."""
    def __init__(self, EPS=0.01, useGPU=False):
        self.EPS = EPS
        self.useGPU = useGPU
        pass

    def parallel_func_evaluation(self, args):
        # print("Received args:", args)
        func, para = args
        return func(para)

    def grad(self, func, para: np.array) -> np.array:
        n = len(para)
        II = np.eye(n)
        plus = para + self.EPS * II
        minus = para - self.EPS * II

        with Pool() as pool:
            plus_eval = pool.map(self.parallel_func_evaluation, [(func, p) for p in plus])
            minus_eval = pool.map(self.parallel_func_evaluation, [(func, m) for m in minus])

        gradient = (np.array(plus_eval) - np.array(minus_eval)) / (2 * self.EPS)

        return gradient

    # def grad_gpu(self, func, para: cp.array) -> cp.array:
    #     para = cp.asarray(para)
    #
    #     n = len(para)
    #     II = cp.eye(n)
    #     plus = para + self.EPS * II
    #     minus = para - self.EPS * II
    #
    #     plus_eval = cp.array([func(p) for p in plus])
    #     minus_eval = cp.array([func(m) for m in minus])
    #
    #     gradient = (plus_eval - minus_eval) / (2 * self.EPS)
    #     return gradient

    def val_shift(self, gradient):
        pass

    def step(self, func, para):
        gradient = self.grad(func, para) if not self.useGPU else self.grad_gpu(func, para)
        para += self.val_shift(gradient)
        para = np.mod(para, 2 * np.pi)
        return para