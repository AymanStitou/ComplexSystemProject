from CascadingFailure_alg_2 import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.read_graphml("degree_distribution/toy_network_undirected.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)


simulation = CascadingFailureSimulation(G)
simulation.calculate_centrality_measures()
simulation.calculate_initial_load(centrality_type='betweenness')
simulation.calculate_capacity(alpha=0.2, beta=1)

initial_failures = [11]
failed_nodes_timestep, _, _, failed_nodes_order = simulation.simulate_cascading_failure(initial_failures)

simulation.animation_network(initial_failures, failed_nodes_timestep, save_anim=True)  # Set save_anim=True to save the animation
