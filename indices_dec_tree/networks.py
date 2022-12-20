import networkx as nx


class NetworkUtils:

    @staticmethod
    def remove_isolates(g):
        isolates = list(nx.isolates(g))
        g.remove_nodes_from(isolates)


class NetworkData:

    @staticmethod
    def get_path_graph(n=8):
        G_path = nx.Graph()
        G_path.add_nodes_from([i for i in range(n)])
        G_path.add_edges_from([(i, i+1) for i in range(n-1)])
        return G_path

    @staticmethod
    def get_triangle_path_graph():
        node_list = [i for i in range(7)]
        edge_list = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 6)]

        G_tpath = nx.Graph()
        G_tpath.add_nodes_from(node_list)
        G_tpath.add_edges_from(edge_list)

        return G_tpath

    @staticmethod
    def get_star_graph(n=6):
        node_list = [i for i in range(n)]
        edge_list = [(0, i) for i in range(1, n)]

        G_star = nx.Graph()
        G_star.add_nodes_from(node_list)
        G_star.add_edges_from(edge_list)

        return G_star

    @staticmethod
    def get_unbalanced_star_graph():
        node_list = [i for i in range(11)]
        edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (0, 8), (8, 9), (0, 10)]

        G_star = nx.Graph()
        G_star.add_nodes_from(node_list)
        G_star.add_edges_from(edge_list)

        return G_star

    @staticmethod
    def get_caterpillar_graph():
        node_list = [i for i in range(8)]
        edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (3, 7)]

        G_5 = nx.Graph()
        G_5.add_nodes_from(node_list)
        G_5.add_edges_from(edge_list)
        
        return G_5

    @staticmethod
    def get_lollipop_graph():
        node_list = [i for i in range(4)]
        edge_list = [(0, 1), (1, 2), (2, 3), (0, 2)]

        G_8 = nx.Graph()
        G_8.add_nodes_from(node_list)
        G_8.add_edges_from(edge_list)

        return G_8

    @staticmethod
    def get_connected_stars():    
        node_list = [i for i in range(10)]
        edge_list = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6), (5, 7), (5, 8), (5, 9)]

        G_10 = nx.Graph()
        G_10.add_nodes_from(node_list)
        G_10.add_edges_from(edge_list)

        return G_10

    @staticmethod
    def get_florentine_families_graph():
        n = 16
        node_list = ['Acciaiuoli', 'Albizzi', 'Barbadori', 'Bischeri', 'Castellani', 'Ginori', 'Guadagni',
                     'Lamberteschi', 'Medici', 'Pazzi', 'Peruzzi', 'Pucci', 'Ridolfi', 'Salviati', 'Strozzi',
                     'Tornabuoni']
        # Business arcs
        #sB = [3,3,3,3,4,4,4,5,5,5,6,6,7,7,8,8,8,8,9,9,9,9,9,10,11,11,11,11,14,16]
        #tB = [5,6,9,11,7,8,11,3,8,11,3,9,4,8,4,5,7,11,3,6,10,14,16,9,3,4,5,8,9,9]
        
        # Marital arcs
        sM = [1,2,2,2,3,3,4,4,4,5,5,5,6,7,7,7,7,8,9,9,9,9,9,9,10,11,11,11,13,13,13,14,14,15,15,15,15,16,16,16]
        tM = [9,6,7,9,5,9,7,11,15,3,11,15,2,2,4,8,16,7,1,2,3,13,14,16,14,4,5,15,9,15,16,9,10,4,5,11,13,7,9,13]

        links = set()
        #for i in range(len(sB)):
        #    links.add((sB[i]-1,tB[i]-1))
        for i in range(len(sM)):
            links.add((sM[i]-1,tM[i]-1))

        edge_list = [(node_list[s], node_list[t]) for s, t in links]

        G_Medici = nx.Graph()
        G_Medici.add_nodes_from(node_list)
        G_Medici.add_edges_from(edge_list)

        G_Medici.remove_node('Pucci') # Pucci - isolated vertex

        return G_Medici

    @staticmethod
    def get_9_11_terrorist_graph():
        # Medici
        n = 19
        node_list = ['Ahmed Alghamdi', 'Hamza Alghamdi', 'Mohand Alshehri', 'Fayez Ahmed', 'Marwan Al-Shehhi',
                     'Ahmed Alnami', 'Saeed Alghamdi', 'Ahmed Al-Haznawi', 'Ziad Jarrah', 'Salem Alhazmi',
                     'Nawaf Alhazmi', 'Khalid Al-Mihdhar', 'Hani Hanjour', 'Majed Moqed', 'Mohamed Atta',
                     'Abdul Aziz Al-Omari', 'Waleed Alshehri', 'Satam Suqami', 'Wail Alshehri']
        links = [(0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (1, 10), (2, 3), (3, 4), (4, 8), (4, 9), (4, 12), (4, 14),
                 (4, 15), (5, 6), (5, 10), (6, 7), (6, 10), (7, 8), (8, 9), (8, 12), (8, 14), (9, 10), (10, 11),
                 (10, 12), (10, 14), (11, 12), (12, 13), (12, 14), (14, 15), (15, 16), (16, 17), (16, 18), (17, 18)]
        edge_list = [(node_list[s], node_list[t]) for s, t in links]

        G_911 = nx.Graph()
        G_911.add_nodes_from(node_list)
        G_911.add_edges_from(edge_list)

        return G_911


    @staticmethod
    def get_sampson_t1_digraph():
        """
        Sampson network
        :return: Directed graph
        """
        edge_list = [(0, 2), (0, 4), (0, 13), (1, 0), (1, 6), (1, 13), (2, 0), (2, 1), (2, 16), (3, 4), (3, 5), (3, 9),
                     (4, 3), (4, 10), (4, 12), (5, 0), (5, 3), (5, 8), (6, 1), (6, 7), (6, 15), (7, 0), (7, 1), (7, 8),
                     (8, 4), (8, 7), (8, 15), (9, 3), (9, 7), (9, 13), (10, 4), (10, 7), (10, 13), (11, 0), (11, 1),
                     (11, 13), (12, 4), (12, 6), (12, 17), (13, 0), (13, 10), (13, 11), (14, 0), (14, 1), (14, 13),
                     (15, 0), (15, 1), (15, 5), (16, 2), (16, 12), (16, 17), (17, 0), (17, 1), (17, 6)]
        G = nx.DiGraph()
        G.add_nodes_from([i for i in range(18)])
        G.add_edges_from(edge_list)

        c_integration = [4.47, 4.24, 3.65, 3.65, 4.24, 3.18, 3.71, 3.88, 3.24, 2.71, 3.71, 3.24, 3.47, 4.18, 0.00, 3.41,
                         2.71, 2.76]
        c_radiality = [3.41, 3.41, 3.29, 3.47, 3.35, 3.41, 3.12, 3.35, 3.41, 3.47, 3.47, 3.18, 3.41, 3.18, 3.41, 3.47,
                       3.18, 3.41]

        return G, c_integration, c_radiality


if __name__ == "__main__":
    import numpy as np
    G = NetworkData.get_lollipop_graph()
    print(G)
