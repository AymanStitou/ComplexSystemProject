import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.animation import PillowWriter

class CascadingFailureSimulation:
    def __init__(self, G=None):
        assert G is not None, "Graph G must be provided."
        self.original_G = G.copy()  
        self.G = G.copy()  
        self.CF = 0
        self.N = len(self.G.nodes)
        for node in self.G.nodes:
            self.G.nodes[node]['load'] = 0
            self.G.nodes[node]['capacity'] = 0
            self.G.nodes[node]['degree_centrality'] = 0
            self.G.nodes[node]['betweenness_centrality'] = 0
            self.G.nodes[node]['closeness_centrality'] = 0

    def calculate_centrality_measures(self):
        """
        Calculates degree, betweenness, and closeness centralities
        and stores them as node attributes.
        """
        degree_centrality = nx.degree_centrality(self.G) 
        betweenness_centrality = nx.betweenness_centrality(self.G, normalized=False)
        closeness_centrality = nx.closeness_centrality(self.G)

        # Multiply degree centrality by (N - 1) to match the typical "degree" count
        for node in self.G.nodes:
            self.G.nodes[node]['degree_centrality'] = degree_centrality[node] * (self.N - 1)
            self.G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
            self.G.nodes[node]['closeness_centrality'] = closeness_centrality[node]

    def calculate_initial_load(self, centrality_type='degree'):
        """
        Sets node load = chosen centrality measure.
        """
        for node in self.G.nodes:
            if centrality_type == 'degree':
                self.G.nodes[node]['load'] = self.G.nodes[node]['degree_centrality']
            elif centrality_type == 'betweenness':
                self.G.nodes[node]['load'] = self.G.nodes[node]['betweenness_centrality']
            elif centrality_type == 'closeness':
                self.G.nodes[node]['load'] = self.G.nodes[node]['closeness_centrality']
            else:
                raise ValueError(f"Unknown centrality type: {centrality_type}")

    def calculate_capacity(self, alpha=0.2, beta=1.5):
        """
        Capacity = (1 + alpha) * (load^beta).
        """
        for node in self.G.nodes:
            load = self.G.nodes[node]['load']
            self.G.nodes[node]['capacity'] = (1 + alpha) * (load ** beta)

    def simulate_cascading_failure(self, initial_failures):
        """
        Initiates cascading failures with given initial failures.
        Tracks failures at each timestep.
        Returns:
            failed_nodes_timestep: list where each element is a set of nodes failed at that timestep
            self.CF: Cascading Failure measure
            I: Impact factor
            failed_nodes_order: list of nodes in the order they failed
        """
        failed_nodes = set(initial_failures)
        failed_nodes_order = list(initial_failures)
        failed_nodes_timestep = [set(initial_failures)] 
        
        # Initial removal of failed nodes
        for node in initial_failures:
            load_to_distribute = self.G.nodes[node]['load']
            neighbors = list(self.G.neighbors(node)) if not self.G.is_directed() else list(self.G.successors(node))
            self.G.remove_node(node)

            if neighbors:
                sum_neighbours_capacity = sum(self.G.nodes[neighbor]['capacity'] for neighbor in neighbors)

                if sum_neighbours_capacity == 0:
                    # Edge case: Distribute load equally
                    extra_load = load_to_distribute / len(neighbors)
                    for neighbor in neighbors:
                        if neighbor not in failed_nodes:
                            self.G.nodes[neighbor]['load'] += extra_load
                            if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                                failed_nodes.add(neighbor)
                                failed_nodes_order.append(neighbor)
                                failed_nodes_timestep[-1].add(neighbor)
                else:
                    # Distribute load proportionally based on capacity
                    for neighbor in neighbors:
                        if neighbor not in failed_nodes:
                            redistributed_load = (load_to_distribute * 
                                                 (self.G.nodes[neighbor]['capacity'] / sum_neighbours_capacity))
                            self.G.nodes[neighbor]['load'] += redistributed_load
                            if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                                failed_nodes.add(neighbor)
                                failed_nodes_order.append(neighbor)
                                failed_nodes_timestep[-1].add(neighbor)

        timestep = 0
        
        # Iteratively process newly failed nodes
        while True:
            new_failures = set()
            for node in failed_nodes_timestep[timestep]:
                if self.G.has_node(node):
                    load_to_distribute = self.G.nodes[node]['load']
                    neighbors = list(self.G.neighbors(node)) if not self.G.is_directed() else list(self.G.successors(node))
                    self.G.remove_node(node)

                    if neighbors:
                        sum_neighbours_capacity = sum(self.G.nodes[neighbor]['capacity'] for neighbor in neighbors)

                        if sum_neighbours_capacity == 0:
                            # Edge case: Distribute load equally
                            extra_load = load_to_distribute / len(neighbors)
                            for neighbor in neighbors:
                                if neighbor not in failed_nodes:
                                    self.G.nodes[neighbor]['load'] += extra_load
                                    if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                                        new_failures.add(neighbor)
                                        failed_nodes_order.append(neighbor)
                        else:
                            # Distribute load proportionally based on capacity
                            for neighbor in neighbors:
                                if neighbor not in failed_nodes:
                                    redistributed_load = (load_to_distribute * 
                                                        (self.G.nodes[neighbor]['capacity'] / sum_neighbours_capacity))
                                    self.G.nodes[neighbor]['load'] += redistributed_load
                                    if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                                        new_failures.add(neighbor)
                                        failed_nodes_order.append(neighbor)
            if not new_failures:
                break

            # Update for next timestep
            failed_nodes_timestep.append(new_failures)
            failed_nodes.update(new_failures)
            timestep += 1

        # Remove empty timesteps if any
        failed_nodes_timestep = [t for t in failed_nodes_timestep if t]

        NA = len(initial_failures)
        total_failed = len(failed_nodes)
        self.CF = NA / (total_failed * self.N) if total_failed > 0 else 0
        I = total_failed / self.N 

        return failed_nodes_timestep, self.CF, I, failed_nodes_order

    def visualize_network(self, failed_nodes):
        """
        Quick utility to visualize failed (red) vs. active (green) nodes.
        """
        pos = nx.spring_layout(self.original_G)
        node_colors = ['red' if node in failed_nodes else 'green' for node in self.original_G.nodes]
        nx.draw(self.original_G, pos, with_labels=True, node_color=node_colors, node_size=500)
        plt.show()

    def print_centrality_measures(self):
        """
        Debugging: prints out node-level centralities.
        """
        print("Node centrality measures:")
        for node in self.G.nodes:
            print(f"Node {node}: Degree={self.G.nodes[node]['degree_centrality']:.2f}, "
                  f"Betweenness={self.G.nodes[node]['betweenness_centrality']:.2f}, "
                  f"Closeness={self.G.nodes[node]['closeness_centrality']:.2f}")

    def animation_network(self, initial_failures, failed_nodes_timestep, save_anim=False): 
        pos = nx.spring_layout(self.original_G)
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "red", "brown"])
        fig, ax = plt.subplots(figsize=(10, 8))

        color_meanings = {
            'red': 'Initial Failures',
            'brown': 'Cascading Failures',
            'green': 'Surviving Nodes'
        }

        # Create a list to keep track of failed nodes up to the current frame
        cumulative_failed = set(initial_failures)

        def update(frame):
            ax.clear()
            if frame == 0: 
                colors = []
                for node in self.original_G.nodes:
                    if node in initial_failures:
                        colors.append('red')
                    else: 
                        colors.append('green')
                nx.draw(self.original_G, pos, ax=ax, with_labels=True, node_color=colors, node_size=800, font_size=10, font_weight='bold')
                ax.set_title(f"Cascading Failures in Nodes - Timestep {frame + 1}")
                legend_elements = [Patch(facecolor=color, edgecolor='black', label=meaning) 
                                   for color, meaning in color_meanings.items()]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.8, 1))
            elif frame <= len(failed_nodes_timestep): 
                newly_failed = failed_nodes_timestep[frame-1]
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
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.8, 1))
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
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.8, 1))
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

        anim = animation.FuncAnimation(fig, update, frames=len(failed_nodes_timestep)+1, interval=1000, repeat=False)

        if save_anim: 
            writer = PillowWriter(fps=1)
            anim.save('network_animation.gif', writer=writer)
        plt.show()

    def rank_centrality(self, centrality_type='degree', length=None): 
        """
            Ranks the nodes in the network based on the centrality.

            Parameters:
            centrality_type (str): The type of centrality to rank.
            length (int): The number of top-ranked nodes to display.
        """

        if centrality_type == 'degree': 
            degree_centralities = {node: self.G.nodes[node]['degree_centrality'] for node in self.G.nodes}
            rank_centrality_results = sorted(degree_centralities.items(), key=lambda x: x[1], reverse=True)
        elif centrality_type == 'betweenness': 
            betweenness_centralities = {node: self.G.nodes[node]['betweenness_centrality'] for node in self.G.nodes}
            rank_centrality_results = sorted(betweenness_centralities.items(), key=lambda x: x[1], reverse=True)
        elif centrality_type == 'closeness': 
            closeness_centralities = {node: self.G.nodes[node]['closeness_centrality'] for node in self.G.nodes}
            rank_centrality_results = sorted(closeness_centralities.items(), key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")

        for i, (node, centrality) in enumerate(rank_centrality_results[:length], 1):
            print(f"{i}: The node {node} has the centrality of {centrality}")
        ranked_nodes = [node for node, centrality in rank_centrality_results]

        return ranked_nodes[:length]
