# Testing new simulation algorithm with toy network


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from CascadingFailure import CascadingFailureSimulation  # Import your class

# Load network from GraphML file
G = nx.read_graphml("replicated network/custom_network.graphml")
mapping = {node: int(node) for node in G.nodes()}  # Ensure node labels are integers
G = nx.relabel_nodes(G, mapping)

# Initialize simulation
simulation = CascadingFailureSimulation(G)

# Compute centrality measures and set initial loads
simulation.calculate_centrality_measures()
simulation.calculate_initial_load(centrality_type="betweenness")

# Range of α values to test
alpha_values = np.linspace(0.0, 1.5, 15)  # Adjust step size as needed
I_values = []  # Store the fraction of failed nodes

# Test cascading failure for different α values
for alpha in alpha_values:
    simulation.calculate_capacity(alpha=alpha, beta=1.2)  # Keep β fixed at 1.2
    failed_nodes, CF, I, failed_nodes_list = simulation.simulate_cascading_failure(initial_failures=[11])
    I_values.append(I)  # Store I for plotting
    print(len(failed_nodes))

# Plot I vs α
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, I_values, marker='o', linestyle='-', color='b', label="Fraction of failed nodes")
plt.xlabel(r'$\alpha$ (Capacity Scaling Parameter)')
plt.ylabel(r'$I$ (Fraction of Failed Nodes)')
plt.title("Cascading Failures: I vs α")
plt.legend()
plt.grid()
plt.show()

# Visualize the network after cascading failure
simulation.visualize_network(failed_nodes)


for node in G.nodes:
        print(f"Node {node}: {G.nodes[node]['betweenness_centrality']:.6f}")