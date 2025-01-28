import logging 
import networkx as nx
import plotting as plot
from scipy.stats import ks_2samp
def get_networkgraph(filepath):
    try:
        #Trying to import and read the save Graph File
        graph = nx.read_graphml(filepath)
        logging.info('Graph Successfully imported')
    except FileNotFoundError:
        logging.error("Error: GraphML file not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading graph: {e}")
        exit(1)

    mapping = {node: int(node) for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)
    logging.info("Graph nodes have been successfully relabeled.")
    logging.info(f"The graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    return graph

def create_ER_network(graph):
    n = graph.number_of_nodes()  
    k = graph.number_of_edges()

    p = 2*k/(n*(n-1)) # Probability for the ER graph to have k edges 
    ER_graph = nx.erdos_renyi_graph(n, p)
    logging.info(f"Erdos–Renyi graph created with {n} nodes and {ER_graph.number_of_edges()} edges.")

    return ER_graph

def compare_degree_distributions(G, ER_graph, plot_dist = True, stat_test = True, prob_dist = True):
    
    # Degree distribution for the original network (G)
    n = G.number_of_nodes()
    G_degrees = [G.degree(node) for node in G.nodes()]
    G_degree_count = {deg: G_degrees.count(deg) for deg in set(G_degrees)}  
    G_degree_prob = {deg: count / n for deg, count in G_degree_count.items()}
    logging.info("Degree distribution for the original network (G) calculated.")

    # Degree distribution for the ER graph
    ER_degrees = [ER_graph.degree(node) for node in ER_graph.nodes()]
    ER_degree_count = {deg: ER_degrees.count(deg) for deg in set(ER_degrees)}
    ER_degree_prob = {deg: count / n for deg, count in ER_degree_count.items()}
    logging.info("Degree distribution for the Erdos-Renyi graph calculated.")

    if plot_dist:
        logging.info('Degree distribution for the ER graph calculated')
        plot.plot_degree_dists(G_degrees, ER_degrees)
    
    if stat_test:
        logging.info('Performing KS test between the original graph and the ER graph')
        ks_stat, p_value = ks_2samp(list(G_degree_prob.values()), list(ER_degree_prob.values()))
        print(f"KS test result between G and ER: statistic={ks_stat}, p-value={p_value}")

    if prob_dist:
        return G_degree_prob, ER_degree_prob
    
def calculate_clustering_coefficient(graph):
    '''Calculte Clustering Coefficient of the graph'''
    return nx.average_clustering(graph)

def calculate_average_path_length(graph):
    '''Calculate average path length of the graph'''
    try:
        return nx.average_shortest_path_length(graph)
    except nx.NetworkXError:
        logging.error("Graph is disconnected, cannot compute average path length.")
        return None

def calculate_degree_assortativity(graph):
    '''Calculate the degree assortativity of the graph'''
    return nx.degree_assortativity_coefficient(graph)

def compare_graph_metrics(graph1, graph2):
    logging.info('Comparing Metric between two graphs')
    graph1_clustering = calculate_clustering_coefficient(graph1)
    graph2_clustering = calculate_clustering_coefficient(graph2)
    print(f"Clustering Coefficient - Network1: {graph1_clustering:0.5f}, Network2: {graph2_clustering:0.5f}")
    
    # Average Path Length
    graph1_avg_path_length = calculate_average_path_length(graph1)
    graph2_avg_path_length = calculate_average_path_length(graph2)
    print(f"Average Path Length - Network1: {graph1_avg_path_length:0.5f}, Network2: {graph2_avg_path_length:0.5f}")
    
    # Degree Assortativity
    graph1_assortativity = calculate_degree_assortativity(graph1)
    graph2_assortativity = calculate_degree_assortativity(graph2)
    print(f"Degree Assortativity - Network1: {graph1_assortativity:0.5f}, Network2: {graph2_assortativity:0.5f}")


