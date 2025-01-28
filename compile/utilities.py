import logging 
import networkx as nx
import plotting as plot
from scipy.stats import ks_2samp
def get_networkgraph(filepath):
    try:
        #Trying to import and read the save Graph File
        G = nx.read_graphml(filepath)
        logging.info('Graph Successfully imported')
    except FileNotFoundError:
        logging.error("Error: GraphML file not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading graph: {e}")
        exit(1)

    mapping = {node: int(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    logging.info("Graph nodes have been successfully relabeled.")
    logging.info(f"The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    return G

def create_ER_network(G):
    n = G.number_of_nodes()  
    k = G.number_of_edges()

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



