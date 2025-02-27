from CascadingFailure import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.read_graphml("replicated network/toy_network_undirected.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)


simulation = CascadingFailureSimulation(G)
simulation.calculate_centrality_measures()
simulation.calculate_initial_load(centrality_type='betweenness')
simulation.calculate_capacity(alpha=0.2, beta=1)

initial_failures = [11]
_, _, _, failed_nodes_list = simulation.simulate_cascading_failure(initial_failures)
simulation.animation_network(initial_failures, failed_nodes_list, save_anim=False) #save_anim=True if you want to save the animation



