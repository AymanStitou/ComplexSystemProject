# CascadingFailure_alg_2.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.animation import PillowWriter
import numpy as np

class CascadingFailureSimulation:
    def __init__(self, G=None):
        assert G is not None, "Graph G must be provided."
        self.original_G = G.copy()  # Preserve the original graph for visualization
        self.G = G.copy()           # Working copy for simulation
        self.CF = 0
        self.N = len(self.G.nodes)
        
        # Initialize node attributes
        for node in self.G.nodes:
            self.G.nodes[node]['load'] = 0
            self.G.nodes[node]['capacity'] = 0
            self.G.nodes[node]['degree_centrality'] = 0
            self.G.nodes[node]['betweenness_centrality'] = 0
            self.G.nodes[node]['closeness_centrality'] = 0
        
        # Precompute and store neighbors for efficiency
        self.original_neighbors = {node: set(self.original_G.neighbors(node)) for node in self.original_G.nodes}
        
        # Create mappings between node IDs and indices for NumPy arrays
        self.node_to_index = {node: idx for idx, node in enumerate(self.original_G.nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}
        
        # Initialize load and capacity arrays using NumPy for performance
        self.node_loads = np.zeros(self.N)
        self.node_capacities = np.zeros(self.N)

    def calculate_centrality_measures(self):
        """
        Calculates degree, betweenness, and closeness centralities
        and stores them as node attributes.
        """
        degree_centrality = nx.degree_centrality(self.G) 
        betweenness_centrality = nx.betweenness_centrality(self.G, normalized=False)
        closeness_centrality = nx.closeness_centrality(self.G)

        # Store centrality measures as node attributes
        for node in self.G.nodes:
            self.G.nodes[node]['degree_centrality'] = degree_centrality[node] * (self.N - 1)
            self.G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
            self.G.nodes[node]['closeness_centrality'] = closeness_centrality[node]

    def calculate_initial_load(self, centrality_type='degree'):
        """
        Sets node load based on the chosen centrality measure.
        """
        for node in self.G.nodes:
            idx = self.node_to_index[node]
            if centrality_type == 'degree':
                self.node_loads[idx] = self.G.nodes[node]['degree_centrality']
            elif centrality_type == 'betweenness':
                self.node_loads[idx] = self.G.nodes[node]['betweenness_centrality']
            elif centrality_type == 'closeness':
                self.node_loads[idx] = self.G.nodes[node]['closeness_centrality']
            else:
                raise ValueError(f"Unknown centrality type: {centrality_type}")

    def calculate_capacity(self, alpha=0.2, beta=1.5):
        """
        Calculates node capacities using the formula:
        Capacity = (1 + alpha) * (load^beta)
        """
        self.node_capacities = (1 + alpha) * (self.node_loads ** beta)

    def simulate_cascading_failure(self, initial_failures):
        """
        Initiates cascading failures with given initial failures.
        Tracks failures at each timestep.
        Returns:
            failed_nodes_timestep: list of sets, each set contains nodes failed at that timestep
            self.CF: Cascading Failure measure
            I: Impact factor (fraction of nodes failed)
            failed_nodes_order: list of nodes in the order they failed
        """
        failed_nodes = set(initial_failures)
        failed_nodes_order = list(initial_failures)
        failed_nodes_timestep = [set(initial_failures)]
        
        # Initialize active nodes (all nodes initially active)
        active_nodes = set(self.original_G.nodes()) - failed_nodes
        
        # References for efficiency
        original_neighbors = self.original_neighbors
        node_loads = self.node_loads
        node_capacities = self.node_capacities
        
        # Process initial failures
        for node in initial_failures:
            idx = self.node_to_index[node]
            load_to_distribute = node_loads[idx]
            neighbors = original_neighbors[node] & active_nodes  # Active neighbors
            
            if neighbors:
                sum_neighbours_capacity = sum(node_capacities[self.node_to_index[neighbor]] for neighbor in neighbors)
                
                if sum_neighbours_capacity == 0:
                    # Distribute load equally among neighbors
                    extra_load = load_to_distribute / len(neighbors)
                    for neighbor in neighbors:
                        if neighbor not in failed_nodes:
                            neighbor_idx = self.node_to_index[neighbor]
                            node_loads[neighbor_idx] += extra_load
                            if node_loads[neighbor_idx] > node_capacities[neighbor_idx]:
                                failed_nodes.add(neighbor)
                                failed_nodes_order.append(neighbor)
                                failed_nodes_timestep[-1].add(neighbor)
                else:
                    # Distribute load proportionally based on capacity
                    for neighbor in neighbors:
                        if neighbor not in failed_nodes:
                            neighbor_idx = self.node_to_index[neighbor]
                            redistributed_load = load_to_distribute * (node_capacities[neighbor_idx] / sum_neighbours_capacity)
                            node_loads[neighbor_idx] += redistributed_load
                            if node_loads[neighbor_idx] > node_capacities[neighbor_idx]:
                                failed_nodes.add(neighbor)
                                failed_nodes_order.append(neighbor)
                                failed_nodes_timestep[-1].add(neighbor)
        
        timestep = 0
        
        # Iteratively process newly failed nodes
        while timestep < len(failed_nodes_timestep):
            new_failures = set()
            for node in failed_nodes_timestep[timestep]:
                neighbors = original_neighbors[node] & active_nodes  # Active neighbors
                if not neighbors:
                    continue  # No active neighbors to redistribute load
                
                idx = self.node_to_index[node]
                load_to_distribute = node_loads[idx]
                sum_neighbours_capacity = sum(node_capacities[self.node_to_index[neighbor]] for neighbor in neighbors)
                
                if sum_neighbours_capacity == 0:
                    # Distribute load equally
                    extra_load = load_to_distribute / len(neighbors)
                    for neighbor in neighbors:
                        if neighbor not in failed_nodes:
                            neighbor_idx = self.node_to_index[neighbor]
                            node_loads[neighbor_idx] += extra_load
                            if node_loads[neighbor_idx] > node_capacities[neighbor_idx]:
                                new_failures.add(neighbor)
                                failed_nodes_order.append(neighbor)
                else:
                    # Distribute load proportionally based on capacity
                    for neighbor in neighbors:
                        if neighbor not in failed_nodes:
                            neighbor_idx = self.node_to_index[neighbor]
                            redistributed_load = load_to_distribute * (node_capacities[neighbor_idx] / sum_neighbours_capacity)
                            node_loads[neighbor_idx] += redistributed_load
                            if node_loads[neighbor_idx] > node_capacities[neighbor_idx]:
                                new_failures.add(neighbor)
                                failed_nodes_order.append(neighbor)
            
            if not new_failures:
                break  # No new failures in this timestep
            
            # Update for next timestep
            failed_nodes_timestep.append(new_failures)
            failed_nodes.update(new_failures)
            active_nodes -= new_failures
            timestep += 1
        
        # Remove empty timesteps if any
        failed_nodes_timestep = [t for t in failed_nodes_timestep if t]
        
        # Calculate metrics
        NA = len(initial_failures)
        total_failed = len(failed_nodes)
        self.CF = NA / (total_failed * self.N) if total_failed > 0 else 0
        I = total_failed / self.N 
        
        return failed_nodes_timestep, self.CF, I, failed_nodes_order

    def visualize_network(self, failed_nodes):
        """
        Visualizes failed (red) vs. active (green) nodes.
        """
        pos = nx.spring_layout(self.original_G, seed=42)  # Fixed seed for consistent layout
        node_colors = ['red' if node in failed_nodes else 'green' for node in self.original_G.nodes]
        nx.draw(self.original_G, pos, with_labels=True, node_color=node_colors, node_size=500)
        plt.show()

    def print_centrality_measures(self):
        """
        Prints out node-level centralities for debugging.
        """
        print("Node centrality measures:")
        for node in self.G.nodes:
            print(f"Node {node}: Degree={self.G.nodes[node]['degree_centrality']:.2f}, "
                  f"Betweenness={self.G.nodes[node]['betweenness_centrality']:.2f}, "
                  f"Closeness={self.G.nodes[node]['closeness_centrality']:.2f}")

    def animation_network(self, initial_failures, failed_nodes_timestep, save_anim=False): 
        """
        Creates an animation showing the progression of cascading failures.
        
        Parameters:
        - initial_failures (list): List of initially failed nodes.
        - failed_nodes_timestep (list of sets): Nodes failed at each timestep.
        - save_anim (bool): Whether to save the animation as a GIF.
        """
        pos = nx.spring_layout(self.original_G, seed=42)  # Fixed seed for consistent layout
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "red", "brown"])
        fig, ax = plt.subplots(figsize=(10, 8))

        color_meanings = {
            'red': 'Initial Failures',
            'brown': 'Cascading Failures',
            'green': 'Surviving Nodes'
        }

        # Keep track of failed nodes up to the current frame
        cumulative_failed = set(initial_failures)

        def update(frame):
            ax.clear()
            if frame < len(failed_nodes_timestep):
                newly_failed = failed_nodes_timestep[frame]
                cumulative_failed.update(newly_failed)
                colors = []
                for node in self.original_G.nodes:
                    if node in initial_failures:
                        colors.append('red')
                    elif node in cumulative_failed:
                        colors.append('brown')
                    else:
                        colors.append('green')
                nx.draw(self.original_G, pos, ax=ax, with_labels=True, node_color=colors, node_size=800, font_size=10, font_weight='bold')
                ax.set_title(f"Cascading Failures in Nodes - Timestep {frame + 1}")
                legend_elements = [Patch(facecolor=color, edgecolor='black', label=meaning) 
                                   for color, meaning in color_meanings.items()]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            else:
                # Final frame
                colors = []
                for node in self.original_G.nodes:
                    if node in initial_failures:
                        colors.append('red')
                    elif node in cumulative_failed:
                        colors.append('brown')
                    else:
                        colors.append('green')
                nx.draw(self.original_G, pos, ax=ax, with_labels=True, node_color=colors, node_size=800, font_size=10, font_weight='bold')
                ax.set_title("Cascading Failures in Nodes - Final State")
                legend_elements = [Patch(facecolor=color, edgecolor='black', label=meaning) 
                                   for color, meaning in color_meanings.items()]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

        anim = animation.FuncAnimation(fig, update, frames=len(failed_nodes_timestep)+1, interval=1000, repeat=False)

        if save_anim: 
            writer = PillowWriter(fps=1)
            anim.save('network_animation.gif', writer=writer)
        plt.show()

    def rank_centrality(self, centrality_type='degree', length=None): 
        """
        Ranks the nodes in the network based on the specified centrality.
        
        Parameters:
        - centrality_type (str): The type of centrality to rank ('degree', 'betweenness', 'closeness').
        - length (int): The number of top-ranked nodes to display.
        
        Returns:
        - ranked_nodes (list): List of top-ranked nodes based on the centrality measure.
        """
        if centrality_type == 'degree': 
            centralities = self.node_loads.copy()
        elif centrality_type in ['betweenness', 'closeness']:
            centralities = np.array([self.G.nodes[node][f'{centrality_type}_centrality'] for node in self.G.nodes])
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")

        # Rank nodes based on centrality (descending order)
        ranked_indices = np.argsort(-centralities)
        ranked_nodes = [self.index_to_node[idx] for idx in ranked_indices]
        ranked_centralities = centralities[ranked_indices]

        # Print top 'length' nodes
        if length is None:
            length = self.N
        for i, (node, centrality) in enumerate(zip(ranked_nodes, ranked_centralities), 1):
            if i > length:
                break
            print(f"{i}: The node {node} has the centrality of {centrality}")
        
        return ranked_nodes[:length]
