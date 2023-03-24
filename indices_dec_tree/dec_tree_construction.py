import numpy as np
import networkx as nx
from .network_generation import *
from .centralities_batch import CentralityBatchAnalyzer, set_values_precision
from collections import defaultdict


def generator_of_general_graphs(geng_path, n0=4, n1=9, nmax=11):
    """
    General procedure enumerating the entire set of connected graphs
    :param geng_path: path to geng utility
    :param n0: step 2 starting graph order
    :param n1: step 3 starting graph order
    :param nmax: reasonable graph order
    :return: graph sequence
    """
    # step 2.1 graphs
    for n in range(n0, n1):
        for g in nauty_geng(geng_path, options=f'{n} {n} -c', debug=False):
            yield g
    # step 2.2 graphs
    for n in range(n0, n1):
        for g in nauty_geng(geng_path, options=f'{n} {n-1} -c', debug=False):
            yield g
    # step 2.3 graphs
    for n in range(n0, n1):
        for g in nauty_geng(geng_path, options=f'{n} {n+1} -c', debug=False):
            yield g
    # step 3.1 graphs
    for n in range(n1, nmax + 1):
        # step 3.1.1
        # (1)-(3)
        for g in nauty_geng(geng_path, options=f'{n} {n} -c', debug=False):
            yield g
        for g in nauty_geng(geng_path, options=f'{n} {n-1} -c', debug=False):
            yield g
        if n+1 <= 2*n-8:
            for g in nauty_geng(geng_path, options=f'{n} {n+1}:{2*n-8} -c', debug=False):
                yield g
        # step 3.1.2 (4)
        for ns in range(n0, n+1):
            if ns+n-7 >= n-1:  # connected condition
                for g in nauty_geng(geng_path, options=f'{n} {ns+n-7} -c', debug=False):
                    yield g


def generator_of_unicycle_trees(n0, nmax, dmax):
    """
    General procedure enumerating the entire set of connected u-trees
    :param n0: min nodes
    :param nmax: max nodes
    :param dmax: max node degree
    :return: u-tree sequence
    """
    for n in range(n0, nmax + 1):
        u_trees = generate_unicyclic_trees(n, max_degree=dmax)
        for u in u_trees:
            yield u


class CentralityPairs:
    """
    Accounting for distinguished measures
    """
    def __init__(self, c_list):
        """
        :param c_list: list of measures
        """
        n = len(c_list)
        self.c_list = c_list
        self.all = {(i, j) for i in range(n) for j in range(i + 1, n)}
        self.distinguished = set()

    def update_distinguished(self, new_distinguished):
        self.distinguished |= new_distinguished

    def get_undistinguished(self, keep_ids=True):
        """
        Set of undistinguished pairs of centrality measures
        :param keep_ids: measures ids
        :return:
        """
        un = self.all - self.distinguished
        if keep_ids:
            un = self.as_id_pairs(un)
        return un

    def as_id_pairs(self, pairs):
        """
        Pairs of measures' identifiers
        :param pairs: set of pairs of measures' indexes
        :return:
        """
        return {(self.c_list[i1].id, self.c_list[i2].id) for i1, i2 in pairs}

    def as_id_pair(self, pair):
        """
        Pair of measures' identifiers
        :param pair: a pair of measures' indexes
        :return:
        """
        i1, i2 = pair
        return self.c_list[i1].id, self.c_list[i2].id


class CentralityDistinguisher:
    """
    Finding distinguishing triples for a list of centralities
    """

    def __init__(self, min_nodes=4, med_nodes=8, max_nodes=11, max_degree=4, geng_path=None, precision=6):
        """
        :param min_nodes: minimum number of test graph nodes
        :param max_nodes: maximum number of test graph nodes
        :param med_nodes: interim number of test graph nodes
        :param max_degree: maximum node degree of the test graph
        :param geng_path: path to geng utility
        :param precision:  rounding of centrality values
        """
        self.min_nodes = min_nodes
        self.med_nodes = med_nodes
        self.max_nodes = max_nodes
        self.max_degree = max_degree
        self.geng_path = geng_path
        self.precision = precision

        self.c_list = CentralityBatchAnalyzer.generate_centralities()
        self.c_pairs = CentralityPairs(self.c_list)
        self.graphs = []
        self.analyzers = []

        # indices of distinguishing graphs
        self.distinguishing_number_to_cc2uvs = defaultdict()
        self.distinguishing_number_to_uv2ccs = defaultdict()

        cnt = self.distinguish_measures()
        assert cnt == 0, f'Failed to distinguish all measures (remained {cnt}: {self.c_pairs.get_undistinguished(keep_ids=True)}), try to increase the max_nodes value'

    def get_distinguishing_node_pairs(self, graph_index, c1, c2):
        """
        Get pairs of vertices that give differences for centralities c1 and c2
        :param graph_index: graph number
        :param c1: measure of centrality
        :param c2: measure of centrality
        :return: pairs of vertices, matrix of differences in the ordering of vertices pairs
        """
        node_list = [n for n in self.graphs[graph_index].nodes()]
        cba = self.analyzers[graph_index]
        c1_values = cba.centrality_to_nvalues[c1]
        c2_values = cba.centrality_to_nvalues[c2]
        vals1 = [c1_values[n] for n in node_list]
        vals2 = [c2_values[n] for n in node_list]

        matrix = np.sign(np.subtract.outer(vals1, vals1)) != np.sign(np.subtract.outer(vals2, vals2))
        indices = np.argwhere(matrix)
        if indices.size == 0:
            return [], matrix
        else:
            return [(node_list[idx[0]], node_list[idx[1]]) for idx in indices if idx[0] < idx[1]], matrix

    def distinguish_measures(self):
        """
        Distinguish measures of centrality
        :return: number of undistinguished measures
        """
        # graphs for distinguishing measures
        self.graphs = []

        # Calculation of centrality measures
        self.distinguishing_number_to_cc2uvs = defaultdict(dict)
        self.distinguishing_number_to_uv2ccs = defaultdict(dict)

        # graph generator
        if self.geng_path is None:
            gen = generator_of_unicycle_trees(self.min_nodes, self.med_nodes, self.max_degree)
        else:
            gen = generator_of_general_graphs(self.geng_path, n0=self.min_nodes, n1=self.med_nodes+1,
                                              nmax=self.max_nodes)
        for number, dgraph in enumerate(gen):
            self.graphs.append(dgraph)

            cba = CentralityBatchAnalyzer(G=dgraph, centralities=self.c_list)
            cba.compute_centralities(precision=self.precision)
            self.analyzers.append(cba)

            for ci1, ci2 in self.c_pairs.all:
                c1 = self.c_list[ci1]
                c2 = self.c_list[ci2]

                node_pairs, m = self.get_distinguishing_node_pairs(number, c1, c2)
                if len(node_pairs) > 0:
                    self.distinguishing_number_to_cc2uvs[number][(c1.id, c2.id)] = node_pairs
                    for uv in node_pairs:
                        if uv not in self.distinguishing_number_to_uv2ccs[number]:
                            self.distinguishing_number_to_uv2ccs[number][uv] = {(c1.id, c2.id), }
                        else:
                            self.distinguishing_number_to_uv2ccs[number][uv].add((c1.id, c2.id))

                    self.c_pairs.update_distinguished({(ci1, ci2), })

            # all measures are distinguished
            if len(self.c_pairs.get_undistinguished()) == 0:
                break

        return len(self.c_pairs.get_undistinguished())


def partition_ccs(ccs, cba, u, v):
    """
    Splitting measures of centrality
    :param ccs: set of measures' identifiers
    :param cba: measures analyzer
    :param u: vertex 1
    :param v: vertex 2
    :return:
    """
    c_ids_partitions = {'>': set(), '<': set(), '=': set()}
    for c_id in ccs:
        c = cba.id_to_centrality[c_id]
        n_vals = cba.centrality_to_nvalues[c]
        if n_vals[u] > n_vals[v]:
            c_ids_partitions['>'].add(c_id)
        elif n_vals[u] < n_vals[v]:
            c_ids_partitions['<'].add(c_id)
        else:
            c_ids_partitions['='].add(c_id)
    return c_ids_partitions


def greedy_distinguishing_G_u_v(cd, un_ccs):
    """
    Find the optimal at the current step G, u, and v
    :param cd: measures distinguisher
    :param un_ccs: undistinguished measures
    :return: index, u, v
    """
    max_grp_sz = 100000
    best_number, best_uv, best_na = None, (None, None), 0

    for graph_number, uv_to_ccs in cd.distinguishing_number_to_uv2ccs.items():
        cba = cd.analyzers[graph_number]
        for uv, ccs in uv_to_ccs.items():
            c_ids_partitions = [len(p_ccs) for p_ccs in partition_ccs(un_ccs, cba, uv[0], uv[1]).values() if len(p_ccs) > 0]
            _max = np.max(c_ids_partitions)
            if _max < max_grp_sz:
                max_grp_sz = _max
                best_uv = uv
                best_number = graph_number
                best_na = len(c_ids_partitions)
            elif _max == max_grp_sz:
                curr_sz = len(cd.graphs[graph_number])
                best_sz = len(cd.graphs[best_number])
                if curr_sz <= best_sz:
                    if (curr_sz < best_sz) or len(c_ids_partitions) > best_na:
                        best_uv = uv
                        best_number = graph_number
                        best_na = len(c_ids_partitions)

    return best_number, best_uv[0], best_uv[1]


def voting_distinguishing_G_u_v(cd, un_ccs):
    """
    Find the optimal G, u, and v at the current step

    :param cd: measures distinguisher
    :param un_ccs: undistinguished measures
    :return: index, u, v
    """
    min_delta = 1.
    best_ratio = 2./3.
    best_number, best_uv, best_na = None, (None, None), 0

    for graph_number, uv_to_ccs in cd.distinguishing_number_to_uv2ccs.items():
        cba = cd.analyzers[graph_number]
        for uv, ccs in uv_to_ccs.items():
            _part_ccs = partition_ccs(un_ccs, cba, uv[0], uv[1])
            _psz = {k: len(v) for k, v in _part_ccs.items()}
            na = np.sum([int(v > 0) for v in _psz.values()])
            ratio = max(_psz['>'], _psz['<']) / np.sum([v for v in _psz.values()])
            delta = abs(best_ratio-ratio)
            if delta < min_delta:
                min_delta = delta
                best_uv = uv
                best_number = graph_number
                best_na = na
            elif delta == min_delta:
                curr_sz = len(cd.graphs[graph_number])
                best_sz = len(cd.graphs[best_number])
                if curr_sz <= best_sz:
                    if (curr_sz < best_sz) or na > best_na:
                        best_uv = uv
                        best_number = graph_number
                        best_na = na

    return best_number, best_uv[0], best_uv[1]


def random_distinguishing_G_u_v(cd, un_ccs):
    """
    Find the optimal G, u, and v at the current step
    :param cd: measures distinguisher
    :param un_ccs: undistinguished measures
    :return: index, u, v
    """
    for graph_number in cd.distinguishing_number_to_cc2uvs:
        for d_c_pair in cd.distinguishing_number_to_cc2uvs[graph_number]:
            u, v = cd.distinguishing_number_to_cc2uvs[graph_number][d_c_pair][0]
            if (d_c_pair[0] in un_ccs) and (d_c_pair[1] in un_ccs):
                break
        else:
            continue
        break
    else:
        assert False, 'The graph distinguishing the measures was not found'
    return graph_number, u, v


def standard_step(tree, cur_node, cd):
    """
    Decision tree construction step
    :param tree: decision tree
    :param cur_node: current node (tree node)
    :param cd: CentralityDistinguisher object
    :return:
    """
    c_ids = tree.nodes[cur_node]['measures']
    if len(c_ids) > 1:
        # Find graph
        # graph_index, u, v = random_distinguishing_G_u_v(cd, c_ids)
        # graph_number, u, v = greedy_distinguishing_G_u_v(cd, c_ids)
        graph_number, u, v = voting_distinguishing_G_u_v(cd, c_ids)

        tree.nodes[cur_node]['graph_index'] = graph_number
        tree.nodes[cur_node]['label'] = f'{u} vs {v} in G_{graph_number}'

        # Split measures: c(u) vs c(v)
        cba = cd.analyzers[graph_number]
        c_ids_partitions = partition_ccs(c_ids, cba, u, v)

        # Construct childs
        for rel, c_ids_partition in c_ids_partitions.items():
            if len(c_ids_partition) > 0:
                child_node = len(tree)
                tree.add_edge(cur_node, child_node, label=f'f({u}) {rel} f({v})', u_rel_v=(u, rel, v))
                tree.nodes[child_node]['measures'] = c_ids_partition
                standard_step(tree, child_node, cd)
    else:
        tree.nodes[cur_node]['label'] = next(iter(c_ids))


def construct_dec_tree(cd):
    """
    Decision tree construction
    :param cd: CentralityDistinguisher object
    :return: decision tree
    """
    tree = nx.DiGraph()
    tree.add_node(0, measures={c.id for c in cd.c_list})
    standard_step(tree, 0, cd)
    return tree

