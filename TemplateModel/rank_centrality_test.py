from CascadingFailure import CascadingFailureSimulation
import networkx as nx

G = nx.read_graphml("replicated network/toy_network_undirected.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)

simulation = CascadingFailureSimulation(G)
simulation.calculate_centrality_measures()
highest_centrality_nodes = simulation.rank_centrality('degree', 5)
print(highest_centrality_nodes)