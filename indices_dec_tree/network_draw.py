import networkx as nx
from IPython.display import Image, display


def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)


def set_gv_nodes_attrs(P):
    node_attrs = {'color': 'lightgray', 'style': 'filled',  # 'shape': 'box',
                  'margin': 0.05,  # 'width': 0, 'height': 0,
                  'fontname': 'Courier', 'fontsize': '14'}
    for n in P.get_nodes():
        n.obj_dict['attributes'].update(node_attrs)    


def _to_custom_pydot(G, settings=None):
    """deprecated"""
    P = nx.nx_pydot.to_pydot(G)

    if settings is not None and isinstance(settings, dict):        
        P.set_graph_defaults(**settings)
    else:
        P.set_graph_defaults(splines='spline', esep=0.5, layout='sfdp', ranksep=1.0, nodesep=0.15)

    set_gv_nodes_attrs(P)

    return P


def to_custom_pydot(G, graph_defaults=None, node_defaults=None, edge_defaults=None, nodes_attrs=None):
    """
    G -> pydot
    """
    
    # gv graph, node, edge defaults
    if graph_defaults is not None and isinstance(graph_defaults, dict):
        G.graph['graph'] = graph_defaults
    else:
        G.graph['graph'] = {'splines': 'spline', 'overlap' : 'scale', 'esep' : 0.5, 'layout': 'neato'}

    if node_defaults is not None and isinstance(node_defaults, dict):
        G.graph['node'] = node_defaults
    else:
        # node_style_none = {'margin': 0, 'width': 0, 'height' :0, 'fontsize': 16, 'shape' : 'none', 'style' : 'filled', 'color': 'lightgray'}
        # color=lightgray, style=filled, fontsize=16
        node_style_box = {'color': 'lightgray', 'style': 'filled',  'fontsize': '16'}
        G.graph['node'] = node_style_box

    # gv edge tooltips
    for e in G.edges():
        G.edges[e[0], e[1]]['tooltip'] = f'{e[0]} - {e[1]}'

    # gv node tooltips
    if nodes_attrs is not None:
        for n in G.nodes():
            node_tooltip = ''
            attrs_vals = nodes_attrs[n]
            for attr in attrs_vals:
                val = attrs_vals[attr]
                if isinstance(val, float):
                    val = round(val, 4)
                node_tooltip += f'{attr} = {val}' + '&#13;&#10;'                
            G.node[n]['tooltip'] = node_tooltip
    
    # convert graph to pydot
    P = nx.nx_pydot.to_pydot(G)
    return P


def draw_as_graphviz(G, settings=None):
    P = to_custom_pydot(G, settings)
    view_pydot(P)


def save_as_graphviz_png(G, path, settings=None):
    P = to_custom_pydot(G, settings)
    P.write_png(path)


def save_as_graphviz_svg(G, path, settings=None):
    P = to_custom_pydot(G, settings)
    P.write_svg(path)