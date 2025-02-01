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

rewired_G = nx.read_graphml("rewire.graphml")
mapping_rewire = {node: int(node) for node in rewired_G.nodes()}
rewired_G = nx.relabel_nodes(rewired_G, mapping_rewire)

# initialize parameters and empty lists
alpha_list = np.linspace(0,1.2,20)
beta_list = np.linspace(1,1.5,20)
alpha_value = 0.3
beta_value = 1.2
centrality_types = ["degree"]

# start simulation of the three networks with varying alpha
results_toy_varing_alpha = simulate_and_average(G, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list)
results_ice_varing_alpha = simulate_and_average(G2, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list)
results_us_varing_alpha = simulate_and_average(G3, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list)
plot_line_graph(results_toy_varing_alpha, beta=beta_value, alpha_list=alpha_list, network_type="Toy Network", file_name="toy_network_test")
plot_line_graph(results_ice_varing_alpha, beta=beta_value, alpha_list=alpha_list, network_type="Iceland Network", file_name="iceland_network")
plot_line_graph(results_us_varing_alpha, beta=beta_value, alpha_list=alpha_list, network_type="US Network", file_name="us_network")

# start simulation of the three networks with varying beta
results_toy_varing_beta = simulate_and_average(G, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list)
results_ice_varing_beta = simulate_and_average(G2, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list)
results_us_varing_beta = simulate_and_average(G3, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list)
plot_line_graph(results_toy_varing_beta, alpha=alpha_value, beta_list=beta_list, network_type="Toy Network", file_name="toy_network")
plot_line_graph(results_ice_varing_beta, alpha=alpha_value, beta_list=beta_list, network_type="Iceland Network", file_name="iceland_network")
plot_line_graph(results_us_varing_beta, alpha=alpha_value, beta_list=beta_list, network_type="US Network", file_name="us_network")

# save the result to .csv file
save_results_to_csv(results_toy_varing_alpha, fr"toy_network_beta_{beta_value}_results.csv", alpha_list)
save_results_to_csv(results_ice_varing_alpha, fr"iceland_network_beta_{beta_value}_results.csv", alpha_list)
save_results_to_csv(results_us_varing_alpha, fr"US_network_beta_{beta_value}_results.csv", alpha_list)
print(f"The result for the toy network with varing alpha is: {results_toy_varing_alpha}")
print(f"The result for the iceland network with varing alpha is: {results_ice_varing_alpha}")
print(f"The result for the US network with varing alpha is: {results_us_varing_alpha}")

save_results_to_csv(results_toy_varing_beta, fr"toy_network_alpha_{alpha_value}_results.csv", beta_list=beta_list)
save_results_to_csv(results_ice_varing_beta, fr"iceland_network_alpha_{alpha_value}_results.csv", beta_list=beta_list)
save_results_to_csv(results_us_varing_beta, fr"US_network_alpha_{alpha_value}_results.csv", beta_list=beta_list)
print(f"The result for the toy network with varing beta is: {results_toy_varing_beta}")
print(f"The result for the iceland network with varing beta is: {results_ice_varing_beta}")
print(f"The result for the US network with varing beta is: {results_us_varing_beta}")