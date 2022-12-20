import networkx as nx
import scipy as sp
import numpy as np
from scipy.sparse import linalg


def laplacian_matrix(A):
    """
    laplacian matrix
    :param A:
    :return:
    """
    n, __ = A.shape
    Diag_d = np.diag(np.squeeze(np.matmul(A, np.ones((n, 1)))))
    return Diag_d - A


def to_stochastic_matrix(M):
    """
    Create row stochastic matrix
    :param M: sparse matrix (scipy.sparse)
    :return:
    """
    M_sum = sp.sparse.diags(1 / M.sum(axis=1).A.ravel())
    return M_sum @ M


def to_sparse_matrix(G):
    """
    Create scipy sparse matrix
    :param G:
    :return:
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Cannot compute centrality for null graph')
    return nx.to_scipy_sparse_matrix(G, nodelist=list(G), weight=None, dtype=float)


def dominant_eig(M, left=False, which='LR'):
    """
    Get eigenvalue with largest real part and right eigenvector (positive & normalized (L2))
    :param M: Sparse matrix (scipy)
    :param left: left eigenvector?
    :return: eigenvalue, eigenvector (numpy.ndarray 1d)
    """
    if left:
        eigenvalue, eigenvector = linalg.eigs(M.T, k=1, which=which, maxiter=150, tol=0)
    else:
        eigenvalue, eigenvector = linalg.eigs(M, k=1, which=which, maxiter=150, tol=0)

    largest = eigenvector.flatten().real
    norm = sp.sign(largest.sum()) * sp.linalg.norm(largest)
    return eigenvalue.real, largest / norm



