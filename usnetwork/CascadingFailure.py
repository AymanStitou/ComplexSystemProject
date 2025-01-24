import networkx as nx
import matplotlib.pyplot as plt

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

        # Multiply degree centrality by (N - 1) so it matches typical "degree" count
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
        Returns:
            failed_nodes: set of all failed nodes
            self.CF: measure as defined in your code
            I: fraction of nodes that failed
        """
        failed_nodes = set(initial_failures)
        queue = list(initial_failures)
        
        while queue:
            node = queue.pop(0)
            # If directed graph, use successors; otherwise use neighbors
            if self.G.is_directed():
                neighbors = list(self.G.successors(node))
            else:
                neighbors = list(self.G.neighbors(node))

            # Sum the capacities of the neighbors
            sum_neighbours = sum([self.G.nodes[neighbor]['capacity'] for neighbor in neighbors])

            if neighbors:
                for neighbor in neighbors:
                    if neighbor not in failed_nodes:
                        if sum_neighbours == 0:
                            # If neighbors have zero capacity (edge case)
                            self.G.nodes[neighbor]['load'] += self.G.nodes[node]['load'] / len(neighbors)
                            failed_nodes.add(neighbor)
                            queue.append(neighbor)
                        else:
                            # Redistribute load proportionally by capacity
                            redistributed_load = (self.G.nodes[node]['load'] *
                                                  (self.G.nodes[neighbor]['capacity'] / sum_neighbours))
                            self.G.nodes[neighbor]['load'] += redistributed_load
                            # Check failure condition
                            if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                                failed_nodes.add(neighbor)
                                queue.append(neighbor)

        NA = len(initial_failures)
        self.CF = NA / (len(failed_nodes) * self.N)
        I = len(failed_nodes) / self.N 

        return failed_nodes, self.CF, I

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
