import logging 
import numpy as np 
import random
import networkx as nx
import plotting as plot
from scipy.stats import ks_2samp
from CascadingFailure import CascadingFailureSimulation
import pandas as pd

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

def create_ER_network(graph, p = 0.2):
    n = graph.number_of_nodes()  

    ER_graph = nx.erdos_renyi_graph(n, p)
    logging.info(f"Erdos-Renyi graph created with {n} nodes and {ER_graph.number_of_edges()} edges.")

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
    print(f"Average Path Length - Network1: {graph1_avg_path_length:0.5f}, Network2: {graph2_avg_path_length}")
    
    # Degree Assortativity
    graph1_assortativity = calculate_degree_assortativity(graph1)
    graph2_assortativity = calculate_degree_assortativity(graph2)
    print(f"Degree Assortativity - Network1: {graph1_assortativity:0.5f}, Network2: {graph2_assortativity:0.5f}")

def create_BA_network(graph, m = 3):
    n = graph.number_of_nodes() 
    BA_graph = nx.barabasi_albert_graph(n, m)
   
    logging.info(f"Barabasi-Albert graph created with {n} nodes and {BA_graph.number_of_edges()} edges.")
    return BA_graph

def create_WS_network(graph, k = 3, p=0.1):
    n = graph.number_of_nodes() 
    WS_graph = nx.watts_strogatz_graph(n, k, p)

    logging.info(f"Watts-Strogatz graph created with {n} nodes and {WS_graph.number_of_edges()} edges.")
    return WS_graph

def run_simulation(graph, alpha, initial_failures, centrality_type, simulation, beta=1):
    """
    Run the cascading failure simulation for a specific network and centrality measure.
    It returns a list of the number of failed nodes for each alpha value.
    """
    n_failed_nodes = []
    I_list = []

    for a in alpha:
        simulation.calculate_initial_load(centrality_type=centrality_type)
        simulation.calculate_capacity(alpha=a, beta=beta)  # Fix beta to 1
        failed_nodes = simulation.simulate_cascading_failure(initial_failures)
        n_failed_nodes.append(len(failed_nodes))
        I_list.append(len(failed_nodes)/len(graph))

    return I_list

def simulate_and_average(graph, alpha, centrality_types, num_simulations=25, beta=1):
    """
    Simulate the cascading failure multiple times and calculate the mean fraction of failed nodes for each centrality type.
    Return a dictionary with centrality measures as keys and mean I_list as values.
    """
    results = {centrality: [] for centrality in centrality_types}
    total_nodes = len(graph.nodes)
    simulation = CascadingFailureSimulation(graph)
    simulation.calculate_centrality_measures()

    for i in range(num_simulations):
        num_failures = max(1, int(total_nodes * 0.01))  # 1% random failures
        initial_failures = random.sample(range(1,total_nodes-1), num_failures)
        
        for centrality in centrality_types:
            I = run_simulation(graph, alpha, initial_failures, centrality, simulation, beta=beta)
            results[centrality].append(I)
    
    # Compute mean I_list for each centrality type across simulations
    mean_results = {centrality: np.mean(results[centrality], axis=0) for centrality in centrality_types}
    return mean_results

def save_results_to_csv(results, alpha, filename):
    """
    Save simulation results to a CSV file.

    """
    df = pd.DataFrame(results)
    df.insert(0, "Alpha", alpha)
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def load_results_from_csv(filename):
    """
    Load simulation results from a CSV file.

    """
    
    df = pd.read_csv(filename)
    
    alpha = df["Alpha"].tolist()
    results = df.drop(columns=["Alpha"]).to_dict(orient="list")
    
    return alpha, results


