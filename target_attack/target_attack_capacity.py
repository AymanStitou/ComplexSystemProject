from utils import simulate_and_average_capacity, plot_line_graph, load_results_from_csv, save_results_to_csv
import networkx as nx
import numpy as np

# load the toy network
G = nx.read_graphml("degree_distribution/toy_network_undirected.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)

# load the iceland network
G2 = nx.read_graphml("degree_distribution/iceland.graphml")
mapping2 = {node: int(node) for node in G2.nodes()}
G2 = nx.relabel_nodes(G2, mapping2)

# load the US network
G3 = nx.read_graphml("degree_distribution/us_network.graphml")
mapping3 = {node: int(node) for node in G3.nodes()}
G3 = nx.relabel_nodes(G3, mapping3)


# initialize parameters and empty lists
capacity_list=np.linspace(20,100,20)
centrality_types = ["degree", "betweenness", "closeness"]

# for test
test_results = simulate_and_average_capacity(G, centrality_types, capacity_list=capacity_list, num_simulations=25, target_attack=True)
save_results_to_csv(test_results, fr"test.csv", capacity_list=capacity_list)