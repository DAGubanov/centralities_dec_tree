from scipy.linalg import expm
import numpy as np
from enum import Enum


def spectral_radius(A):
    r = np.max(np.abs(np.linalg.eigvals(A)))
    return r


def walk_kernel(A, alpha=0.1):
    """
    A : (n x n) weighted adjacency matrix
    alpha : discounting factor ( 0 < alpha < 1/spectral_radius(A) )
    return K - (n x n) kernel matrix
    """
    (n, m) = A.shape
    if n != m:
        raise Exception('The adjacency matrix is not squared.')

    if not np.array_equal(A.T, A):
        raise Exception('The adjacency matrix is not symmetric.')

    inv_spectral_radius = 1.0/spectral_radius(A)
    if (alpha < 0) or (alpha > inv_spectral_radius):        
        raise Exception(f'The discounting factor is outside the valid range (should be in (0, {inv_spectral_radius}) ).')
        
    I = np.eye(n)
    K = np.linalg.matrix_power(I - alpha*A, -1)
    return K


def safe_walk_kernel(A, alpha=1.0):
    alpha = 1.0/(spectral_radius(A) + 1.0/alpha)
    return walk_kernel(A, alpha)


def exponential_diffusion_kernel(A, alpha=1.0):
    """
    Computes exponential diffusion kernel matrix
    - A : (n x n) weighted adjacency matrix
    - alpha : discounting parameter (> 0)
    - return K - (n x n) exponential diffusion kernel matrix
    """
    # squared matrix
    (n, m) = A.shape
    if n != m:
        raise Exception('The adjacency matrix is not squared.')
    
    # symmetric matrix
    if not np.array_equal(A.T, A):
        raise Exception('The adjacency matrix is not symmetric.')
        
    # alpha
    if alpha <= 0:        
        raise Exception(f'The discounting factor is outside the valid range (should be > 0).')

    # Exponential diffusion kernel matrix 
    K = expm(alpha*A)
    
    return K


def heat_kernel(A, alpha=1.0):
    """
    Computes Laplacian exponential diffusion kernel matrix
    - A : (n x n) weighted adjacency matrix
    - alpha : discounting parameter (> 0)
    - return K - (n x n) Laplacian exponential diffusion kernel
    """
    # squared matrix
    (n, m) = A.shape
    if n != m:
        raise Exception('The adjacency matrix is not squared.')
    
    # symmetric matrix / graph is undirected
    if not np.array_equal(A.T, A):
        raise Exception('The adjacency matrix is not symmetric.')
        
    # alpha
    if alpha <= 0:        
        raise Exception(f'The discounting factor is outside the valid range (should be > 0).')

    # Degree Matrix
    Diag_d = np.diag(np.squeeze(np.matmul(A, np.ones((n, 1)))))

    # Laplacian Matrix 
    L = Diag_d - A

    # Laplacian Exponential diffusion kernel matrix 
    K = expm(-alpha*L)
    
    return K    


def forest_kernel(A, alpha=1.0):
    """
    Computes regularized Laplacian kernel matrix
    - A : (n x n) weighted adjacency matrix
    - alpha : discounting parameter (>0)
    - return K - (n x n) regularized Laplacian kernel matrix
    """
    # squared matrix
    (n, m) = A.shape
    if n != m:
        raise Exception('The adjacency matrix is not squared.')
    
    # symmetric matrix / graph is undirected
    if not np.array_equal(A.T, A):
        raise Exception('The adjacency matrix is not symmetric.')
        
    # alpha
    if alpha <= 0:        
        raise Exception(f'The discounting factor is outside the valid range (should be > 0).')

    # Degree Matrix
    Diag_d = np.diag(np.squeeze(np.matmul(A, np.ones((n, 1)))))

    # Laplacian Matrix 
    L = Diag_d - A

    # regularized Laplacian kernel matrix
    K = np.linalg.matrix_power(np.eye(n) + alpha*L, -1)
    
    return K


class Kernel:
    """
    Calculate matrix kernel
    """

    class Category(Enum):
        """
        Kernel types
        """
        WALK = ('walk', safe_walk_kernel)
        COMM = ('comm', exponential_diffusion_kernel)
        HEAT = ('heat', heat_kernel)
        FOREST = ('forest', forest_kernel)

        def __init__(self, label, k_func):
            self.label = label
            self.k_func = k_func

    def __init__(self, ker_type, k_log=False, k_alpha=1.0):
        """
        Create kernel
        :param ker_type: kernel category
        :param k_log: log(K)?
        :param k_alpha: kernel parameter
        """
        self.ker_type = ker_type
        self.k_log = k_log
        self.k_alpha = k_alpha

    def compute(self, A):
        """
        Compute kernel for adjacency matrix
        :param A: adjacency matrix
        :return: kernel matrix
        """
        K = self.ker_type.k_func(A, self.k_alpha)
        if self.k_log:
            K = np.log(K)
        return K

    def get_label(self):
        if self.k_log:
            return f'l_{self.ker_type.label}'
        else:
            return self.ker_type.label
