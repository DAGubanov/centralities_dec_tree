from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import networkx as nx
from scipy.linalg import expm
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .kernels import Kernel, spectral_radius
from .kernel_distances import KernelDistance
from .lin_utils import to_stochastic_matrix, to_sparse_matrix, dominant_eig, laplacian_matrix


def get_adj_matrix(G):
    """
    Get adjacency matrix
    :param G: graph
    :return: adjacency matrix
    """
    return nx.to_numpy_array(G)  # np.asarray(nx.to_numpy_matrix(G))


def create_centrality_dict(G, cc):
    """
    Create dictionary: node id -> centrality value
    :param G: graph
    :param cc: centrality vector
    :return: centrality dict (node_id, centrality_value)
    """
    c_dict = dict()
    for i, nid in enumerate(G.nodes()):
        c_dict[nid] = cc[i]
    return c_dict


def create_centrality_vec(G, c_dict):
    n = G.number_of_nodes()
    c = np.zeros(n)
    for i, nid in enumerate(G.nodes()):
        c[i] = c_dict[nid]
    return c


class Centrality:
    name = None
    default_params = None

    @property
    def id(self):
        """
        Unique identifier: name + non-default parameter values
        :return:
        """
        return self._id

    def _set_parameters(self, params):
        """
        Set centrality parameters
        :param params:
        :return: non-default parameters dictionary
        """
        self._params = self.default_params.copy()
        if not params:
            return None

        non_def_params = dict()
        for p in params:
            if p not in self.default_params:
                raise Exception(f'Unknown parameter {p} for {self.name} centrality.')
            elif params[p] != self.default_params[p]:
                non_def_params[p] = params[p]

        if len(non_def_params) > 0:
            self._params.update(non_def_params)
            return non_def_params
        else:
            return None

    def _set_id(self, non_def_params):
        """
        Set centrality id
        :param non_def_params: non-empty dict
        :return:
        """
        self._id = self.name
        if not non_def_params:
            return

        keys = sorted(non_def_params.keys())
        s = ','.join([f'{k}={non_def_params[k]}' for k in keys])
        self._id = f'{self.name} [{s}]'

    def __init__(self, params=None):
        non_def_params = self._set_parameters(params)
        self._set_id(non_def_params)

    @abstractmethod
    def compute(self, G):
        """
        Compute centrality for graph
        :param G: graph
        :return: centrality dict
        """
        return None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False


class GraphBasedCentrality(Centrality):
    @abstractmethod
    def compute(self, G):
        pass


class CentralityIntegrationRadiality(GraphBasedCentrality):
    """
    "Additive" version of closeness (Integration/Radiality)
    Reference: "Integration and radiality: measuring the extent of an individual's connectedness and reachability
    in a network", doi: 10.1016/s0378-8733(97)00007-5
    """
    name = 'integration/r'
    default_params = {
    }

    @staticmethod
    def centrality_integration_radiality_general(G, method='radiality'):
        """
        doi: 10.1016/s0378-8733(97)00007-5
        :param G: digraph
        :param method: integration (in ties) / radiality (out ties)
        :return: centrality vector
        """
        D = np.asarray(nx.floyd_warshall_numpy(G))
        if method == 'radiality':
            D_ = D
        elif method == 'integration':
            D_ = D.T
        else:
            raise Exception(f'Method {method} is not supported.')

        n, _ = D_.shape
        c = np.zeros(n)
        max_D = np.max(np.where(D_ == np.inf, -np.inf,
                                D_))
        for i in range(n):
            for j in range(n):
                if i != j and D_[i, j] != np.inf:
                    c[i] += 1 + max_D - D_[i, j]
        c = c / (n - 1)
        return c

    @staticmethod
    def centrality_integration_radiality_simple(D):
        """
        Version for simple graph (unweighted, undirected graph containing no graph loops or multiple edges)
        doi: 10.1016/s0378-8733(97)00007-5
        :param D: distance matrix
        :return: centrality vector
        """
        n, _ = D.shape
        return 1 + np.max(D) - np.sum(D, axis=1) / (n - 1)

    def compute(self, G):
        D = np.asarray(nx.floyd_warshall_numpy(G))
        cc = self.centrality_integration_radiality_simple(D)
        return create_centrality_dict(G, cc)


class CentralityPMeans(GraphBasedCentrality):
    """
    p-means centrality
    doi: 10.1016/j.cnsns.2018.08.002
    """
    name = 'p-means'
    default_params = {
        'p': 1.0
    }

    def compute(self, G):
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        p = self._params['p']
        D_ = D.copy()
        if p == 0:
            D_[D_ == 0] = 1
            cc = np.prod(D_, axis=1) ** (-1 / (n - 1))
        else:
            np.power(D, p, out=D_, where=D > 0)
            cc = (np.sum(D_, axis=1) / (n - 1)) ** (-1 / p)

        return create_centrality_dict(G, cc)


class CentralityHarmonicCloseness(GraphBasedCentrality):
    """
    Harmonic version of the closeness centrality
    doi: 10.1080/15427951.2013.865686
    """
    name = 'clo harmonic'
    default_params = {
    }

    def compute(self, G):
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        D_ = D.copy()
        np.divide(1, D, out=D_, where=D > 0)
        cc = np.sum(D_, axis=1) / (n - 1)
        return create_centrality_dict(G, cc)


class CentralityWeightedDegree(GraphBasedCentrality):
    """
    Weighted Degree Centrality
    https://arxiv.org/abs/1703.07580
    """
    name = 'deg weighted'
    default_params = {
    }

    def compute(self, G):
        A = get_adj_matrix(G)
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        D_ = D.copy()
        deg = np.sum(A, axis=0, keepdims=True)
        np.divide(deg, D, out=D_, where=D > 0)
        cc = np.sum(D_, axis=1)  # / (n-1)
        return create_centrality_dict(G, cc)


class CentralityDecayingDegree(GraphBasedCentrality):
    """
    Decaying Degree Centrality
    https://arxiv.org/abs/1703.07580
    """
    name = 'deg decaying'
    default_params = {
    }

    def compute(self, G):
        A = get_adj_matrix(G)
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        D_ = D.copy()
        deg = np.sum(A, axis=0, keepdims=True)
        np.divide(deg, np.power(n, 2 * D), out=D_)
        cc = np.sum(D_, axis=1)  # / (n-1)
        return create_centrality_dict(G, cc)


class CentralityDecay(GraphBasedCentrality):
    """
    Decay Centrality
    https://arxiv.org/abs/1604.05582
    """
    name = 'decay'
    default_params = {
        'delta': 1.0
    }

    def compute(self, G):
        delta = self._params['delta']
        D = np.asarray(nx.floyd_warshall_numpy(G))
        cc = np.sum(np.power(delta, D), axis=1) - 1  # -1 for diagonal elements
        return create_centrality_dict(G, cc)


class CentralitySeeley(GraphBasedCentrality):
    """
    Seeley Centrality (doi: 10.1037/h0084096)
    """
    name = 'seeley'
    default_params = {
    }

    def compute(self, G):
        M = to_stochastic_matrix(to_sparse_matrix(G))
        eigval, eigvec = dominant_eig(M, left=True)
        if np.round(eigval, 6) != 1.0:
            raise ValueError(f'Seeley centrality: eigenvalue ({eigval}) must be equal to 1')
        return create_centrality_dict(G, eigvec)


class CentralityEigenOnDissim(GraphBasedCentrality):
    """
    Eigencentrality based on dissimilarity (doi: 10.1038/srep17095)
    """
    name = 'eig_dissim'
    default_params = {
        'metric': 'jaccard'   # 'dice'
    }

    def compute(self, G):
        A = get_adj_matrix(G)
        n, _ = A.shape
        A = np.asarray(A + np.eye(n), dtype=bool)
        D = squareform(pdist(A, self._params['metric']))
        W = csr_matrix(A * D)
        eigval, eigvec = dominant_eig(W, left=False)
        return create_centrality_dict(G, eigvec)


class CentralityBetaCurrentFlow(GraphBasedCentrality):
    """
    Beta Current Flow Centrality
    doi: 10.1007/978-3-319-21786-4_19
    """
    name = 'bCF'
    default_params = {
        'beta': 1.0
    }

    def compute(self, G):
        A = nx.to_numpy_array(G)  # np.asarray(nx.to_numpy_matrix(G))
        n, __ = A.shape
        L = laplacian_matrix(A)

        phi = np.linalg.matrix_power(np.eye(n) * self._params['beta'] + L, -1)

        y = np.zeros(n)
        for i in range(n):
            for k in range(n):
                s = 0.0
                for j in range(n):
                    s += np.abs(phi[i, j] - phi[k, j])
                y[i] += A[i, k] * s
        y = (y + 1) / (2 * n)

        return create_centrality_dict(G, y)


class CentralityBridging(GraphBasedCentrality):
    """
    Bridging Centrality:
    Identifying Bridging Nodes In Scale-free Networks
    https://ubir.buffalo.edu/xmlui/bitstream/handle/10477/34552/2006-05.pdf
    """
    name = 'bridging'
    default_params = {
    }

    def compute(self, G):
        A = nx.to_numpy_array(G)  # np.asarray(nx.to_numpy_matrix(G))
        n, __ = A.shape
        deg = np.sum(A, axis=1)

        BC = 1 / (deg * A.dot(1 / deg))
        bc_dict = dict(zip(G, BC))
        cr_dict = nx.betweenness_centrality(G)
        for k in cr_dict:
            cr_dict[k] = cr_dict[k] * bc_dict[k]
        return cr_dict


class CentralityEstrada(GraphBasedCentrality):
    """
    Estrada centrality
    doi: 10.1103/PhysRevE.71.056103
    """
    name = 'estrada'
    default_params = {
    }

    def compute(self, G):
        A = nx.to_numpy_array(G)  # np.asarray(nx.to_numpy_matrix(G))
        cc = np.diag(expm(A))
        return create_centrality_dict(G, cc)


class CentralityTotalComm(GraphBasedCentrality):
    """
    Total communicability
    doi: 10.1093/comnet/cnt007
    """
    name = 'total_comm'
    default_params = {
    }

    def compute(self, G):
        A = nx.to_numpy_array(G)  # np.asarray(nx.to_numpy_matrix(G))
        cc = np.sum(expm(A), axis=1)
        return create_centrality_dict(G, cc)


class CentralityBetweenness(GraphBasedCentrality):
    name = 'bet'
    default_params = {
        'normalized': False
    }

    def compute(self, G):
        c_dict = nx.betweenness_centrality(G, **self._params)
        return c_dict


class CentralityEccentricity(GraphBasedCentrality):
    name = 'ecc'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.eccentricity(G)
        for k in c_dict:
            c_dict[k] = 1.0 / c_dict[k]
        return c_dict


class CentralityDegree(GraphBasedCentrality):
    name = 'deg'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.degree_centrality(G, **self._params)
        return c_dict


class CentralityCloseness(GraphBasedCentrality):
    name = 'clo'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.closeness_centrality(G, **self._params)
        return c_dict


class CentralityEccentricity(GraphBasedCentrality):
    name = 'ecc_base'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.eccentricity(G, **self._params)
        c_dict = {n: 1./v for n, v in c_dict.items()}
        return c_dict


class CentralityCoreness(GraphBasedCentrality):
    """
    Coreness centrality
    """
    name = 'coreness'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.core_number(G, **self._params)
        return c_dict


class CentralityPagerank(GraphBasedCentrality):
    name = 'pr'
    default_params = {
        'alpha': 0.85, 'personalization': None, 'max_iter': 1000, 'tol': 1e-07, 'dangling': None
    }

    def compute(self, G):
        c_dict = nx.pagerank(G, **self._params)
        return c_dict


class CentralityEigenvector(GraphBasedCentrality):
    """
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html
    """

    name = 'eig'
    default_params = {
        'max_iter': 500
    }

    def compute(self, G):
        c_dict = nx.eigenvector_centrality(G, **self._params)
        return c_dict


class CentralityGeneralizedDegree(GraphBasedCentrality):
    name = 'gen_deg'
    default_params = {
        'alpha': 2.0
    }

    def compute(self, G):
        alpha = self._params['alpha']
        A = get_adj_matrix(G)
        n = A.shape[0]
        I = np.eye(n)
        d = np.sum(A > 0, axis=0)

        Diag_d = np.diag(np.squeeze(np.matmul(A, np.ones((n, 1)))))
        L = Diag_d - A

        K = np.linalg.matrix_power(I + alpha * L, -1)
        cc = np.matmul(K, d)
        return create_centrality_dict(G, cc)


class CentralityBonacich(GraphBasedCentrality):
    name = 'bonacich'
    default_params = {
        'alpha': 2.0
    }

    def compute(self, G):
        alpha = self._params['alpha']
        A = get_adj_matrix(G)
        n = A.shape[0]
        I = np.eye(n)
        d = np.sum(A > 0, axis=0)

        alpha = 1.0 / (spectral_radius(A) + 1.0 / alpha)
        K = np.linalg.matrix_power(I - alpha * A, -1)
        cc = np.matmul(K, d)
        return create_centrality_dict(G, cc)


class CentralityKatz(GraphBasedCentrality):
    name = 'katz'
    default_params = {
        'alpha': 1.0, 'beta': 1.0, 'normalized': False
    }

    def compute(self, G):
        alpha = self._params['alpha']
        beta = self._params['beta']
        normalized = self._params['normalized']

        A = get_adj_matrix(G)
        if alpha is None:
            alpha = 1.0 / (spectral_radius(A) + 1.0)
        else:
            alpha = 1.0 / (spectral_radius(A) + 1.0 / alpha)
        return nx.katz_centrality(G, alpha=alpha, beta=beta, normalized=normalized)


def cpm_node(X, r):
    """
    Connectivity power measure (CPM)
    :param X: adjacency matrix (with 0s in the main diagonal)
    :param r: index of a node (from 0 to N-1) in the graph
    :return:
    """
    X = X.copy()
    n = X.shape[0]
    np.fill_diagonal(X, 0)
    nodes_ids = np.arange(n) + 1
    neis_ids = X[:, r].flatten() * nodes_ids

    if np.sum(neis_ids) == 0:
        return 1

    X[:, r] = 0
    X[r, :] = 0
    n_comps, nodes_comps = connected_components(csgraph=csr_matrix(X), directed=False, return_labels=True)
    comps_vals = np.zeros(n_comps)

    for comp_i in range(n_comps):
        if comp_i == nodes_comps[r]:
            comps_vals[comp_i] = 0
        else:
            comp_nodes_ids = (nodes_comps == comp_i) * nodes_ids
            comp_sz = np.sum(comp_nodes_ids > 0)

            if (comp_sz == 1) and (np.any(neis_ids == np.nonzero(comp_nodes_ids)[0][0]+1)):
                comps_vals[comp_i] = 1
            else:
                z_old_neis_idlist = neis_ids[(neis_ids > 0) & (comp_nodes_ids > 0)] - 1
                z_old_idlist = comp_nodes_ids[comp_nodes_ids > 0] - 1
                Z = X[np.ix_(z_old_idlist, z_old_idlist)]

                for i, _id1 in enumerate(z_old_neis_idlist):
                    new_id1 = np.nonzero(z_old_idlist == _id1)[0][0]
                    for _id2 in z_old_neis_idlist[i+1:]:
                        new_id2 = np.nonzero(z_old_idlist == _id2)[0][0]
                        Z[new_id1, new_id2] = 1
                        Z[new_id2, new_id1] = 1
                    comps_vals[comp_i] += cpm_node(Z, new_id1)
                    Z = np.delete(Z, new_id1, axis=0)
                    Z = np.delete(Z, new_id1, axis=1)
                    z_old_idlist = z_old_idlist[z_old_idlist != _id1]

    comps_vals += 1
    return np.prod(comps_vals)


class CentralityConnectednessPower(GraphBasedCentrality):
    name = 'conn_pow'
    default_params = {}

    def compute(self, G):
        X = nx.to_numpy_array(G)  # np.asarray(nx.to_numpy_matrix(G))
        n = X.shape[0]
        c = np.zeros(n)
        for i in range(n):
            c[i] = cpm_node(X, i)
        return create_centrality_dict(G, c)


##### CONNECTIVITY POWER

import scipy.special

def compute_all_binom(max_n):
    """
    Calculation of binomial coefficients
    :param max_n: maximum size
    :return: matrix of binomial coefficients
    """
    n = max_n
    binom_n_k = np.zeros((n,n))
    for c in range(n):
        for r in range(n-c):
            binom_n_k[n-c-1,r] = scipy.special.binom(n-c, r+1)
    return binom_n_k


def conn_degree_vertex(X, r, binom_n_k):
    """
    Calculate connectivity degree
    :param X: adjacency matrix
    :param r: selected vertex
    :param binom_n_k: precomputed binomial coefficients
    :return:
    """
    X = X.copy()
    np.fill_diagonal(X, 0)
    n = X.shape[0]
    c = 1

    if n != 1:
        multinom_params = []
        r_adj = X[:, r].flatten()
        X[:, r] = 0
        X[r, :] = 0
        n_comps, vertex_comp = connected_components(csgraph=csr_matrix(X), directed=False, return_labels=True)

        for comp_i in range(n_comps):
            if comp_i != vertex_comp[r]:
                comp_vertices = []
                comp_vertices_sz = 0
                neis_indices_in_comp = []
                for v in range(n):
                    if vertex_comp[v] == comp_i:
                        comp_vertices.append(v)
                        comp_vertices_sz += 1
                        if r_adj[v] > 0:
                            neis_indices_in_comp.append(comp_vertices_sz-1)
                multinom_params.append(comp_vertices_sz)
                if comp_vertices_sz > 1:
                    Z = X[np.ix_(comp_vertices, comp_vertices)]
                    for k, new_neis_i in enumerate(neis_indices_in_comp):
                        for new_neis_j in neis_indices_in_comp[k+1:]:
                            Z[new_neis_i, new_neis_j] = 1
                            Z[new_neis_j, new_neis_i] = 1
                    c_h_sum = 0
                    for new_neis_k in neis_indices_in_comp:
                        c_k = conn_degree_vertex(Z, new_neis_k, binom_n_k)
                        c_h_sum = c_h_sum + c_k
                    c = c * c_h_sum

        # compute multinomial coefficients
        n = np.sum(multinom_params)
        k = 0
        multinomial = 1
        for i in range(len(multinom_params)):
            n -= k
            k = multinom_params[i]
            multinomial = multinomial * binom_n_k[n-1,k-1]

        c = c * multinomial
    return c


def conn_degree_graph(A):
    """
    Calculate connectivity degree
    A - adj. matrix
    """
    n = A.shape[0]
    binom_n_k = compute_all_binom(n)
    c = np.zeros(n)
    for i in range(n):
        c[i] = conn_degree_vertex(A, i, binom_n_k)
    return c


class CentralityConnectivity(GraphBasedCentrality):

    name = 'conn'
    default_params = {}

    def compute(self, G):
        M = nx.to_numpy_array(G)  # np.asarray(nx.to_numpy_matrix(G))
        c = conn_degree_graph(M)
        return create_centrality_dict(G, c)


class KernelBasedCentrality(Centrality):
    k_type = None
    name = None
    default_params = {
        'k_log': False,
        'k_a': 1.0,
    }

    def _set_id(self, non_def_params):
        """
        Set centrality id
        :param non_def_params: non-empty dict
        :return:
        """
        self._id = self.name
        if self._params['k_log']:
            self._id = f'l_{self.name}'

        if not non_def_params:
            return
        elif 'k_a' in non_def_params:
            self._id += f' [a={non_def_params["k_a"]}]'

    def __init__(self, params=None):
        super().__init__(params)
        self.kernel = Kernel(self.k_type, self._params['k_log'], self._params['k_a'])

    @abstractmethod
    def compute(self, G):
        pass


class CentralityCommKii(KernelBasedCentrality):
    k_type = Kernel.Category.COMM
    name = f'{k_type.label} kii'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.diag(K)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityWalkKii(KernelBasedCentrality):
    k_type = Kernel.Category.WALK
    name = f'{k_type.label} kii'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.diag(K)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityCommKij(KernelBasedCentrality):
    k_type = Kernel.Category.COMM
    name = f'{k_type.label} kij'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.sum(K - np.diag(np.diag(K)), axis=1)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityWalkKij(KernelBasedCentrality):
    k_type = Kernel.Category.WALK
    name = f'{k_type.label} kij'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.sum(K - np.diag(np.diag(K)), axis=1)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class DistanceBasedCentrality(Centrality):

    class CType(Enum):
        CLOSENESS = 'clo'
        ECCENTRICITY = 'ecc'

        def __init__(self, label):
            self.label = label

    c_type = None
    k_type = None
    name = None

    default_params = {
        'k_log': False,
        'k_a': 1.0,
        'd_squared': False,
        'd_norm_func_name': None
    }

    def _set_id(self, non_def_params):
        """
        Set centrality id
        :param non_def_params: non-empty dict
        :return:
        """
        self._id = self.c_type.label
        if self._params['k_log']:
            self._id += f' l_{self.k_type.label}'
        else:
            self._id += f' {self.k_type.label}'

        if not non_def_params:
            return
        else:
            keys = sorted(non_def_params.keys())
            if 'k_log' in keys:
                keys.remove('k_log')
            if len(keys) > 0:
                s = ','.join([f'{k}={non_def_params[k]}' for k in keys])
                self._id += f' [{s}]'

    def __init__(self, params=None):
        super().__init__(params)
        self.kernel = Kernel(self.k_type, self._params['k_log'], self._params['k_a'])
        self.distance = KernelDistance(self.kernel, self._params['d_squared'], self._params['d_norm_func_name'])

    def compute_distance(self, G):
        D = self.distance.compute(get_adj_matrix(G))
        return D

    def compute(self, G):
        D = self.distance.compute(get_adj_matrix(G))
        cc = None
        if self.c_type == self.CType.CLOSENESS:
            (n, _) = D.shape
            D_sum = np.matmul(D, np.ones((n, 1)))
            cc = (n - 1) / np.squeeze(D_sum)
        elif self.c_type == self.CType.ECCENTRICITY:
            cc = 1.0 / np.max(D, axis=1)

        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityClosenessComm(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.COMM
    name = f'{c_type.label} {k_type.label}'


class CentralityClosenessForest(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.FOREST
    name = f'{c_type.label} {k_type.label}'


class CentralityClosenessHeat(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.HEAT
    name = f'{c_type.label} {k_type.label}'


class CentralityClosenessWalk(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.WALK
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityComm(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.COMM
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityForest(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.FOREST
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityHeat(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.HEAT
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityWalk(DistanceBasedCentrality):
    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.WALK
    name = f'{c_type.label} {k_type.label}'



