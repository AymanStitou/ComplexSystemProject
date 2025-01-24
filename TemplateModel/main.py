from CascadingFailure import CascadingFailureSimulation
import networkx as nx

G = nx.read_graphml("replicated network/custom_network.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)


simulation = CascadingFailureSimulation(G)


simulation.calculate_centrality_measures()
simulation.print_centrality_measures()

simulation.calculate_initial_load(centrality_type="degree")


simulation.calculate_capacity(alpha=0.2, beta=1.5)


initial_failures = [11]  
failed_nodes, CF = simulation.simulate_cascading_failure(initial_failures)
print("CF:", CF)

simulation.visualize_network(failed_nodes)


print("Failed nodes:", failed_nodes)