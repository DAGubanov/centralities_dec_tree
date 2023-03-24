import shlex
import subprocess
import networkx as nx
import numpy as np


def nonisomorphic_graphs(graphs):
    """
    Filtering isomorphic graphs
    :param graphs: list of graphs
    :return: list of non-isomorphic graphs
    """
    noniso_graphs = []
    for i, G in enumerate(graphs):
        for j in range(i):
            G_prev = graphs[j]
            if nx.is_isomorphic(G, G_prev):
                break
        else:
            noniso_graphs.append(G)
    return noniso_graphs


def generate_unicyclic_trees(n, max_degree=4):
    """
    Generating 1-trees

    :param n: 1-tree size
    :param max_degree: maximal degree
    :return: list of 1-trees
    """
    trees = list(nx.nonisomorphic_trees(n, create="graph"))
    u_trees = []
    for t in trees:
        # add edge
        for i in range(n):
            for j in range(i + 1, n):
                if not t.has_edge(i, j):
                    ut = t.copy()
                    ut.add_edge(i, j)
                    _degrees = np.array([ut.degree[n] for n in ut.nodes()])
                    if not (np.all(_degrees == 2)) and (np.max(_degrees) <= max_degree):
                        u_trees.append(ut)
    return nonisomorphic_graphs(u_trees)


# NAUTY NON-ISO: http://users.cecs.anu.edu.au/~bdm/nauty/ + Sage helpers (https://www.sagemath.org/)
def _length_and_string_from_graph6(s):
    """
    Return a pair ``(length, graph6_string)`` from a graph6 string of unknown length.
    :param s: a graph6 string describing a binary vector (and encoding its length).
    :return: pair (length, graph6_string)
    """
    if s[0] == chr(126):  # first four bytes are N
        a = format(ord(s[1]) - 63, 'b').zfill(6)
        b = format(ord(s[2]) - 63, 'b').zfill(6)
        c = format(ord(s[3]) - 63, 'b').zfill(6)
        n = int(a + b + c, 2)
        s = s[4:]
    else:  # only first byte is N
        o = ord(s[0])
        if o > 126 or o < 63:
            raise RuntimeError("String seems corrupt: valid chars are \n" + ''.join(chr(i) for i in range(63, 127)))
        n = o - 63
        s = s[1:]
    return n, s


def _binary_string_from_graph6(s):
    """
    Decode a binary string from its graph6 representation
    :param s: a graph6 string
    :return:
    """
    lst = []
    for i in range(len(s)):
        o = ord(s[i])
        if o > 126 or o < 63:
            raise RuntimeError("String seems corrupt: valid chars are \n" + ''.join(chr(j) for j in range(63, 127)))
        a = format(o - 63, 'b')
        lst.append('0'*(6 - len(a)) + a)
    return "".join(lst)


def _from_graph6(g6_string):
    """
    Create ``G`` with the data of a graph6 string.
    Example: G = from_graph6(g, 'IheA@GUAo')
    :param g6_string: a graph6 string
    :return: Graph
    """
    # if isinstance(g6_string, bytes):
    #    g6_string = bytes_to_str(g6_string)
    if not isinstance(g6_string, str):
        raise ValueError('if input format is graph6, then g6_string must be a string')
    n = g6_string.find('\n')
    if n == -1:
        n = len(g6_string)
    ss = g6_string[:n]
    n, s = _length_and_string_from_graph6(ss)
    m = _binary_string_from_graph6(s)
    expected = n * (n - 1) // 2 + (6 - n * (n - 1) // 2) % 6
    if len(m) > expected:
        raise RuntimeError("the string (%s) seems corrupt: for n = %d, the string is too long" % (ss, n))
    elif len(m) < expected:
        raise RuntimeError("the string (%s) seems corrupt: for n = %d, the string is too short" % (ss, n))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    k = 0
    for i in range(n):
        for j in range(i):
            if m[k] == '1':
                G.add_edge(i, j)
            k += 1
    return G


def nauty_geng(geng_path, options="", debug=False):
    """
    Generate non-isomorphic graphs
    :param geng_path: path to geng utility
    :param options: geng options
    :param debug:
    :return:
    """
    sp = subprocess.Popen(shlex.quote(geng_path) + " {0}".format(options), shell=True,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, close_fds=True,
                          encoding='latin-1')
    msg = sp.stderr.readline()
    if debug:
        yield msg
    elif msg.startswith('>E'):
        raise ValueError('wrong format of parameter option')
    gen = sp.stdout
    while True:
        try:
            s = next(gen)
        except StopIteration:
            # Exhausted list of graphs from nauty geng
            return
        yield _from_graph6(s[:-1])
