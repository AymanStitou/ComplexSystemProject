from CascadingFailure import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
# G = nx.read_graphml("replicated network/toy_network_undirected.graphml")
# mapping = {node: int(node) for node in G.nodes()}
# G = nx.relabel_nodes(G, mapping)
G = nx.read_graphml("usnetwork/us_network.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)
total_nodes = len(G.nodes())
simulation = CascadingFailureSimulation(G)
# initial_failures =[random.randint(0, total_nodes-1) for _ in range(10)]

# simulation.calculate_centrality_measures()
# simulation.print_centrality_measures()

# simulation.calculate_initial_load(centrality_type="degree")


# simulation.calculate_capacity(alpha=0.2, beta=1.5)


# initial_failures = [11]  
# failed_nodes, CF = simulation.simulate_cascading_failure(initial_failures)
# print("CF:", CF)

# simulation.visualize_network(failed_nodes)


# print("Failed nodes:", failed_nodes)



simulation.calculate_centrality_measures()


alpha = [0,0.005,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
alpha2 = np.linspace(0,0.25,25)
alpha3 = np.linspace(0.2,0.6,25)
beta = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

n_fail_nodes = []
CF_list = []
I_list = []
for a in alpha3:
    # simulation.calculate_centrality_measures()
    print("->", a)
    simulation.calculate_initial_load(centrality_type='betweenness')
    simulation.calculate_capacity(alpha=a, beta=1)

    initial_failures = [11, 87, 300, 17, 99, 2987, 999, 2222, 55, 43, 22, 99] 
    failed_nodes, CF, I , failed_nodes_list= simulation.simulate_cascading_failure(initial_failures, use_prevention=True)
    n_fail_nodes.append(len(failed_nodes))
    I_list.append(I)
    CF_list.append(CF)

    #simulation.visualize_network(failed_nodes)

# plot CF vs beta/alpha
plt.scatter(alpha3, CF_list, color = "red")
plt.plot(alpha3, CF_list, color = "blue")
plt.xlabel("value of alpha")
plt.ylabel("CF")
plt.title("How CF changes with the value of alhpa")
plt.grid()
plt.show()

# plot I vs beta/alpha
plt.scatter(alpha3, I_list, color = "red")
plt.plot(alpha3, I_list, color = "blue")
plt.xlabel("value of alpha")
plt.ylabel("I")
plt.title("How I changes with the value of alpha")
plt.grid()
plt.show()


# simulation.visualize_network(failed_nodes)


print("Failed nodes:", failed_nodes)
print(CF_list)
# simulation.print_centrality_measures()


