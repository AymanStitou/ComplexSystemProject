from CascadingFailure import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt 
import random
import numpy as np 
G = nx.read_graphml("usnetwork/us_network.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)

alpha = np.linspace(0,1.2,10)
CF_list= []
n_failed_nodes= []
total_nodes = len(G.nodes)
initial_failures =[random.randint(0, total_nodes-1) for _ in range(1)]
simulation = CascadingFailureSimulation(G)
for a in alpha :
    

    simulation.calculate_centrality_measures()
    # simulation.print_centrality_measures()

    simulation.calculate_initial_load(centrality_type="degree")


    simulation.calculate_capacity(alpha=a, beta=1.5)

 
    failed_nodes, CF = simulation.simulate_cascading_failure(initial_failures)
    print("CF:", CF)
    CF_list.append(CF)
    n_failed_nodes.append(len(failed_nodes))
# simulation.visualize_network(failed_nodes)
I= [x/ total_nodes for x in n_failed_nodes]
plt.scatter(alpha, I, color = "red")
plt.plot(alpha, I, color = "blue")
plt.xlabel("value of alpha")
plt.ylabel("I")
plt.title("Plot of I vs alpha for beta = 1.2 and number of nodes: {:.2f}".format(len(initial_failures)))
plt.grid()
plt.show()

# print("Failed nodes:", failed_nodes)