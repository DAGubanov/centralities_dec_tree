import networkx as nx
import numpy as np
from scipy.stats import rankdata, spearmanr
#from centralities import base  # testing mode
from .centralities import base  # notebook mode


def set_values_precision(c_dict, precision=None):
    """
    Round dict values
    :param c_dict:
    :param precision:
    :return: dict
    """
    if precision is not None:
        for n_id in c_dict:
            c_dict[n_id] = round(c_dict[n_id], precision)


class CentralityBatchAnalyzer:
    """
    Manage graph centrality configurations and results
    """
    def __init__(self, G, centralities=None):
        """
        :param G: graph
        :param centralities: centralities list
        """
        self.G = G
        self.A = np.asarray(nx.to_numpy_matrix(G))
        self.centralities = list(centralities) if centralities else self.generate_centralities()
        self.id_to_centrality = {c.id: c for c in self.centralities}
        self.centrality_to_nvalues = dict.fromkeys(self.centralities, {})

    @staticmethod
    def generate_centralities():
        """
        Generate centralities list
        :return:
        """
        c_list = [
            base.CentralityBetweenness(),
            base.CentralityBonacich(),
            base.CentralityDegree(),
            base.CentralityGeneralizedDegree(),
            base.CentralityEigenvector(),
            base.CentralityKatz(),
            base.CentralityPagerank(),
            base.CentralityConnectivity(),
            base.CentralityConnectednessPower(),
            base.CentralityCloseness(),
            base.CentralityEccentricity(),
            base.CentralityCoreness(),

            #base.CentralityIntegrationRadiality(),  # distance-based ~ closeness
            base.CentralityPMeans({'p': -2}),
            base.CentralityPMeans({'p': 0}),
            base.CentralityPMeans({'p': 2}),
            base.CentralityHarmonicCloseness(),
            base.CentralityWeightedDegree(),
            base.CentralityDecayingDegree(),
            base.CentralityDecay({'delta': 0.9}),

            # base.CentralitySeeley(),  # eigenvector-based ~ seeley
            base.CentralityBetaCurrentFlow(),
            base.CentralityBridging(),
            base.CentralityEstrada(),
            base.CentralityTotalComm(),
            base.CentralityEigenOnDissim(),  # jaccard
            base.CentralityEigenOnDissim({'metric': 'dice'}),

            base.CentralityClosenessComm({'k_log': True, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityClosenessForest({'k_log': False, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityClosenessForest({'k_log': True, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityClosenessHeat({'k_log': False, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityClosenessHeat({'k_log': True, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityClosenessWalk({'k_log': True, 'k_a': 1.0, 'd_squared': False}),

            base.CentralityEccentricityComm({'k_log': True, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityEccentricityForest({'k_log': False, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityEccentricityForest({'k_log': True, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityEccentricityHeat({'k_log': False, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityEccentricityHeat({'k_log': True, 'k_a': 1.0, 'd_squared': False}),
            base.CentralityEccentricityWalk({'k_log': True, 'k_a': 1.0, 'd_squared': False}),

            #base.CentralityCommKii({'k_log': False, 'k_a': 1.0}),  # ~ estrada
            base.CentralityCommKij({'k_log': False, 'k_a': 1.0}),
            base.CentralityWalkKii({'k_log': False, 'k_a': 1.0}),
            base.CentralityWalkKij({'k_log': False, 'k_a': 1.0}),
        ]
        return c_list

    @property
    def centrality_identifiers(self):
        return [c.id for c in self.centralities]

    def compute_centralities(self, precision=None):
        """
        Compute all centralities
        :param precision: calculations precision -> tolerance to small fluctuations -> centrality rankings stability
        :return:
        """
        for c in self.centralities:
            c_values = c.compute(self.G)
            set_values_precision(c_values, precision=precision)
            self.centrality_to_nvalues[c] = c_values

    def get_centrality_ranking(self, c):
        """
        Centrality inverse order (decreasing value)
        :param c: centrality (computer)
        :return: ranking list [(key, tiedrank), .. ] decreasing value
        """
        sorted_c = sorted(
            self.centrality_to_nvalues[c].items(),
            key=lambda v: v[1], reverse=True)

        ids, values = zip(*sorted_c)
        ranks = rankdata(values)

        return list(zip(ids, ranks))

    def get_rankings(self):
        """
        Node's ranks
        :return: [c1_ranking_items, c2_ranking_items, ...]
        """
        return [self.get_centrality_ranking(c) for c in self.centralities], self.centrality_identifiers

    def get_scores(self):
        """
        Centrality values
        :return: [node_identifiers, c1_scores, c2_scores, ...]
        """
        scores = [[k for k in self.G.nodes()]]
        for c in self.centralities:
            scores.append([self.centrality_to_nvalues[c][k] for k in self.G.nodes()])
        return scores, self.centrality_identifiers

    def get_correlation_matrix(self):
        """
        Get correlations (spearman) between centralities
        :param method: spearman or pearson
        :return:
        """
        scores, _ = self.get_scores()
        return spearmanr(scores[1:], axis=1).correlation

