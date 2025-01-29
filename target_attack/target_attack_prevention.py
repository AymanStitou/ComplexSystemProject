from utils import simulate_and_average, plot_line_graph, load_results_from_csv, save_results_to_csv
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
alpha_list = np.linspace(0,1.2,20)
beta_list = np.linspace(1,1.5,20)
alpha_value = 0.3
beta_value = 1.2
centrality_types = ["degree", "betweenness", "closeness"]

# simulate three networks with varying alpha, using prevention
results_toy_varing_alpha_prevention = simulate_and_average(G, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list, use_prevention=True)
results_ice_varing_alpha_prevention = simulate_and_average(G2, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list, use_prevention=True)
results_us_varing_alpha_prevention = simulate_and_average(G3, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list, use_prevention=True)
plot_line_graph(results_toy_varing_alpha_prevention, beta=beta_value, alpha_list=alpha_list, network_type="Toy Network with Prevention", file_name="toy_network_prevention")
plot_line_graph(results_ice_varing_alpha_prevention, beta=beta_value, alpha_list=alpha_list, network_type="Iceland Network with Prevention", file_name="iceland_network_prevention")
plot_line_graph(results_us_varing_alpha_prevention, beta=beta_value, alpha_list=alpha_list, network_type="US Network with Prevention", file_name="us_network_prevention")

save_results_to_csv(results_toy_varing_alpha_prevention, fr"toy_network_beta_{beta_value}_prevention_results.csv", alpha_list)
save_results_to_csv(results_ice_varing_alpha_prevention, fr"iceland_network_beta_{beta_value}_prevention_results.csv", alpha_list)
save_results_to_csv(results_us_varing_alpha_prevention, fr"US_network_beta_{beta_value}_prevention_results.csv", alpha_list)

# start simulation of the three networks with varying beta, using prevention
results_toy_varing_beta_prevention = simulate_and_average(G, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list, use_prevention=True)
results_ice_varing_beta_prevention = simulate_and_average(G2, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list, use_prevention=True)
results_us_varing_beta_prevention = simulate_and_average(G3, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list, use_prevention=True)
plot_line_graph(results_toy_varing_beta_prevention, alpha=alpha_value, beta_list=beta_list, network_type="Toy Network with Prevention", file_name="toy_network_prevention")
plot_line_graph(results_ice_varing_beta_prevention, alpha=alpha_value, beta_list=beta_list, network_type="Iceland Network with Prevention", file_name="iceland_network_prevention")
plot_line_graph(results_us_varing_beta_prevention, alpha=alpha_value, beta_list=beta_list, network_type="US Network with Prevention", file_name="us_network_prevention")

save_results_to_csv(results_toy_varing_beta_prevention, fr"toy_network_alpha_{alpha_value}_prevention_results.csv", beta_list=beta_list)
save_results_to_csv(results_ice_varing_beta_prevention, fr"iceland_network_alpha_{alpha_value}_prevention_results.csv", beta_list=beta_list)
save_results_to_csv(results_us_varing_beta_prevention, fr"US_network_alpha_{alpha_value}_prevention_results.csv", beta_list=beta_list)