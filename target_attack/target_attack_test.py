from CascadingFailure import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


G = nx.read_graphml("usnetwork/us_network.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)
total_nodes = len(G.nodes())
num_targeted_nodes = int(0.01*total_nodes)

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


alpha_list = np.linspace(0,1.2,20)

n_fail_nodes = []
CF_list = []
I_list = []
for a in alpha_list:
    # simulation.calculate_centrality_measures()
    print("->", a)
    simulation.calculate_initial_load(centrality_type='degree')
    simulation.calculate_capacity(alpha=a, beta=1.2)

    initial_failures = simulation.rank_centrality(centrality_type='degree', length=num_targeted_nodes) 
    failed_nodes, CF, I , failed_nodes_list= simulation.simulate_cascading_failure(initial_failures, use_prevention=False)
    n_fail_nodes.append(len(failed_nodes))
    I_list.append(I)
    CF_list.append(CF)

    #simulation.visualize_network(failed_nodes)

# plot CF vs beta/alpha
# plt.scatter(alpha_list, CF_list, color = "red")
# plt.plot(alpha_list, CF_list, color = "blue")
# plt.xlabel("value of alpha")
# plt.ylabel("CF")
# plt.title("How CF changes with the value of alhpa")
# plt.grid()
# plt.show()

# plot I vs beta/alpha
plt.scatter(alpha_list, I_list, color = "red")
plt.plot(alpha_list, I_list, color = "blue")
plt.xlabel("value of alpha")
plt.ylabel("I")
plt.title("How I changes with the value of alpha")
plt.grid()
plt.show()
plt.savefig('targeted_attack_us')

# simulation.visualize_network(failed_nodes)


print("Failed nodes:", failed_nodes)
print(CF_list)
# simulation.print_centrality_measures()




