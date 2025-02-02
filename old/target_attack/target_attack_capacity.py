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
toy_capacity_list = np.linspace(61,110,40)
iceland_capacity_list = np.linspace(407,600,40)
us_capacity_list = np.linspace(13200, 30000, 10)
centrality_types = ["degree", "betweenness", "closeness"]

toy_results = simulate_and_average_capacity(G, centrality_types, capacity_list=toy_capacity_list, target_attack=True)
plot_line_graph(toy_results, network_type="Toy Network, with Target Attack", capacity_list=toy_capacity_list, file_name="toy_capacity_target")
save_results_to_csv(toy_results, fr"toy_capacity_target.csv", capacity_list=toy_capacity_list)

iceland_results = simulate_and_average_capacity(G2, centrality_types, capacity_list=iceland_capacity_list, target_attack=True)
plot_line_graph(iceland_results, network_type="Iceland Network, with Target Attack", capacity_list=iceland_capacity_list, file_name="ice_capacity_target")
save_results_to_csv(iceland_results, fr"ice_capacity_target.csv", capacity_list=iceland_capacity_list)

us_results = simulate_and_average_capacity(G3, centrality_types, capacity_list=us_capacity_list, target_attack=True)
plot_line_graph(us_results, network_type="US Network, with Target Attack", capacity_list=us_capacity_list, file_name="us_capacity_target")
save_results_to_csv(us_results, fr"us_capacity_target.csv", capacity_list=us_capacity_list)
