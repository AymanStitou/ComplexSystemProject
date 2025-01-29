import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.animation import PillowWriter

class CascadingFailureSimulation:
    def __init__(self, G=None):
        assert G is not None, "Graph G must be provided."
        self.G = G
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
        sum_degree_centrality = 0
        sum_betweenness_centrality = 0
        sum_closeness_centrality = 0

        # Multiply degree centrality by (N - 1) so it matches typical "degree" count
        for node in self.G.nodes:
            self.G.nodes[node]['degree_centrality'] = degree_centrality[node] * (self.N - 1)
            self.G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
            self.G.nodes[node]['closeness_centrality'] = closeness_centrality[node]
            sum_degree_centrality += self.G.nodes[node]['degree_centrality']
            sum_betweenness_centrality += self.G.nodes[node]['betweenness_centrality']
            sum_closeness_centrality += self.G.nodes[node]['closeness_centrality']
        return [sum_degree_centrality, sum_betweenness_centrality, sum_closeness_centrality]

    def calculate_initial_load(self, centrality_type='degree', sum_centrality=None):
        """
        Sets node load = chosen centrality measure.
        """
        if sum_centrality: 
            sum_degree_centrality, sum_betweenness_centrality, sum_closeness_centrality = sum_centrality
            for node in self.G.nodes:
                if centrality_type == 'degree':
                    self.G.nodes[node]['load'] = self.G.nodes[node]['degree_centrality']
                elif centrality_type == 'betweenness':
                    self.G.nodes[node]['load'] = self.G.nodes[node]['betweenness_centrality'] * sum_degree_centrality/sum_betweenness_centrality
                elif centrality_type == 'closeness':
                    self.G.nodes[node]['load'] = self.G.nodes[node]['closeness_centrality'] * sum_degree_centrality/sum_closeness_centrality
                else:
                    raise ValueError(f"Unknown centrality type: {centrality_type}")
        else: 
            for node in self.G.nodes:
                if centrality_type == 'degree':
                    self.G.nodes[node]['load'] = self.G.nodes[node]['degree_centrality']
                elif centrality_type == 'betweenness':
                    self.G.nodes[node]['load'] = self.G.nodes[node]['betweenness_centrality']
                elif centrality_type == 'closeness':
                    self.G.nodes[node]['load'] = self.G.nodes[node]['closeness_centrality']
                else:
                    raise ValueError(f"Unknown centrality type: {centrality_type}")

    def calculate_capacity(self, alpha=0.2, beta=1.5, total_capacity=None):
        """
        Capacity = (1 + alpha) * (load^beta).
        """
        sum_capacity = 0
        for node in self.G.nodes:
            load = self.G.nodes[node]['load']
            self.G.nodes[node]['capacity'] = (1 + alpha) * (load ** beta)
            sum_capacity += self.G.nodes[node]['capacity']
        if total_capacity: 
            print("sum capacity before scaling:", sum_capacity)
            for node in self.G.nodes:
                self.G.nodes[node]['capacity'] *= total_capacity/sum_capacity

    def prevent_cascading_failure(self, failed_nodes):
        affected_neighbors = set()
        for f_node in failed_nodes:
            if self.G.is_directed():
                affected_neighbors.update(self.G.successors(f_node))
            else:
                affected_neighbors.update(self.G.neighbors(f_node))

        affected_neighbors -= failed_nodes

        for node in affected_neighbors:
            if self.G.is_directed():
                neighbors = list(self.G.successors(node))
            else:
                neighbors = list(self.G.neighbors(node))

            valid_neighbors = [
                n for n in neighbors
                if n not in failed_nodes and not any(f in self.G.neighbors(n) for f in failed_nodes)
            ]

            total_load = self.G.nodes[node]['load']

            for neighbor in valid_neighbors:
                available_capacity = self.G.nodes[neighbor]['capacity'] - self.G.nodes[neighbor]['load']

                if available_capacity > 0:
                    redistributed_load = min(available_capacity, total_load / len(valid_neighbors))
                    self.G.nodes[neighbor]['load'] += redistributed_load
                    total_load -= redistributed_load

                if total_load <= 0:
                    break

            self.G.nodes[node]['load'] = total_load


    # def prevent_cascading_failure(self, failed_nodes):
    #     """
    #         This one has a score system. I don't see an improvement, but I only tested a few times.
    #     """
    #     affected_neighbors = set()
    #     for f_node in failed_nodes:
    #         if self.G.is_directed():
    #             affected_neighbors.update(self.G.successors(f_node))
    #         else:
    #             affected_neighbors.update(self.G.neighbors(f_node))

    #     affected_neighbors -= failed_nodes

    #     for node in affected_neighbors:
    #         if self.G.is_directed():
    #             neighbors = list(self.G.successors(node))
    #         else:
    #             neighbors = list(self.G.neighbors(node))

    #         valid_neighbors = [
    #             n for n in neighbors
    #             if n not in failed_nodes and not any(f in self.G.neighbors(n) for f in failed_nodes)
    #         ]

    #         scored_neighbors = sorted(
    #             valid_neighbors,
    #             key=lambda n: (self.G.nodes[n]['capacity'] - self.G.nodes[n]['load']) / (len(list(self.G.neighbors(n))) or 1),
    #             reverse=True
    #         )

    #         total_load = self.G.nodes[node]['load']

    #         for neighbor in scored_neighbors:
    #             available_capacity = self.G.nodes[neighbor]['capacity'] - self.G.nodes[neighbor]['load']

    #             if available_capacity > 0:
    #                 redistributed_load = min(available_capacity, total_load / len(scored_neighbors))
    #                 self.G.nodes[neighbor]['load'] += redistributed_load
    #                 total_load -= redistributed_load

    #             if total_load <= 0:
    #                 break

    #         self.G.nodes[node]['load'] = total_load


    def simulate_cascading_failure(self, initial_failures, use_prevention=False):
        failed_nodes = set(initial_failures)
        queue = list(initial_failures)
        failed_nodes_list = list()

        while queue:
            node = queue.pop(0)
            if self.G.is_directed():
                neighbors = list(self.G.successors(node))
            else:
                neighbors = list(self.G.neighbors(node))

            sum_neighbours = sum([self.G.nodes[neighbor]['capacity'] for neighbor in neighbors])

            if neighbors:
                for neighbor in neighbors:
                    if neighbor not in failed_nodes:
                        if sum_neighbours == 0:
                            self.G.nodes[neighbor]['load'] += self.G.nodes[node]['load'] / len(neighbors)
                            failed_nodes.add(neighbor)
                            queue.append(neighbor)
                            failed_nodes_list.append(neighbor)
                        else:
                            redistributed_load = (self.G.nodes[node]['load'] *
                                                (self.G.nodes[neighbor]['capacity'] / sum_neighbours))
                            self.G.nodes[neighbor]['load'] += redistributed_load
                            if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                                failed_nodes.add(neighbor)
                                queue.append(neighbor)
                                failed_nodes_list.append(neighbor)

            if use_prevention:
                self.prevent_cascading_failure(failed_nodes)

        NA = len(initial_failures)
        self.CF = NA / (len(failed_nodes) * self.N)
        I = len(failed_nodes) / self.N 

        return failed_nodes, self.CF, I, failed_nodes_list

    def visualize_network(self, failed_nodes):
        """
        Quick utility to visualize failed (red) vs. active (green) nodes.
        """
        pos = nx.spring_layout(self.G)
        node_colors = ['red' if node in failed_nodes else 'green' for node in self.G.nodes]
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=500)
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

    def animation_network(self, initial_failures, failed_nodes_list, save_anim=False): 
        pos = nx.spring_layout(self.G)
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "red"])
        fig, ax = plt.subplots(figsize=(10, 8))

        color_meanings = {
            'red': 'Attacked Nodes',
            'brown': 'Failed Nodes',
            'green': 'Surviving Nodes'
        }

        def update(frame):
            ax.clear()
            colors = []
            for i in range(self.N):
                if i in initial_failures:
                    colors.append('red')
                elif i in failed_nodes_list[:frame]:
                    colors.append('brown')
                else:
                    colors.append('green')
            nx.draw(self.G, pos, ax=ax, with_labels=True, node_color=colors, node_size=800, font_size=10, font_weight='bold')
            
            ax.set_title("Cascading Failures in Nodes")
            legend_elements = [Patch(facecolor=color, edgecolor='black', label=meaning) 
                            for color, meaning in color_meanings.items()]
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.8, 1))
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())

        anim = animation.FuncAnimation(fig, update, frames=len(failed_nodes_list)+1, interval=500, repeat=False)

        if save_anim: 
            writer = PillowWriter(fps=2)
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
        
        # for i, (node, centrality) in enumerate(rank_centrality_results[:length], 1):
        #     print(f"{i}: The node {node} has the centrality of {centrality}")
        ranked_nodes = [node for node, centrality in rank_centrality_results]
        
        return ranked_nodes[:length]

