import numpy as np
from .kernels import Kernel


def dist_norm_func_chebotarev(D):
    """
    Norm distance matrix
    """
    D = D * (D.shape[0]-1) / np.trace(D, 1)
    return D


class KernelDistance:
    """
    Create distance matrix
    """

    dist_norm_funcs = {
        'chebotarev': dist_norm_func_chebotarev
    }

    def __init__(self, kernel, squared=False, dist_norm_func_name=None):
        """
        :param kernel: kernel function (class)
        :param squared: is distance squared?
        """
        self.kernel = kernel
        self.squared = squared

        if dist_norm_func_name is not None:
            if dist_norm_func_name in KernelDistance.dist_norm_funcs:
                self.dist_norm_func = KernelDistance.dist_norm_funcs[dist_norm_func_name]
            else:
                raise Exception('Unknown distance norm function')
        else:
            self.dist_norm_func = None

    def compute_for_kernel_matrix(self, K):
        """
        Compute the distance matrix
        - K - (n x n) kernel matrix
        - D - (n x n) distance matrix
        """
        (n, _) = K.shape
        e = np.ones((n, 1))
        diag_K = np.diag(K).reshape((n, 1))
        D = np.matmul(diag_K, e.T) + np.matmul(e, diag_K.T) - 2 * K

        if not self.squared:
            D = np.power(D, 0.5)
        if self.dist_norm_func:
            D = self.dist_norm_func(D)

        return D

    def compute(self, A):
        """
        Computes the distance matrix
        :param A: (n x n) adjacency matrix
        :return: D - (n x n) distance matrix
        """
        K = self.kernel.compute(A)
        if self.kernel.ker_type == Kernel.Category.FOREST and self.kernel.k_log and self.kernel.k_alpha != 1:
            S = (self.kernel.k_alpha-1) * K / np.log(self.kernel.k_alpha)
        else:
            S = K
        D = self.compute_for_kernel_matrix(S)
        return D


