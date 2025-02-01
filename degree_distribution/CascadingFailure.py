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

    def simulate_cascading_failure(self, initial_failures, use_prevention="None"):
        assert all(node in self.G for node in initial_failures), "Error: One or more initial failure nodes are not in the graph!"
        
        failed_nodes = set(initial_failures)
        failed_nodes_list = list(initial_failures)
        LS1 = set(initial_failures)  # Nodes currently failing
        LS2 = set()  # Nodes that have already failed

        if use_prevention == "localized_capacity_boost":
            self.localized_capacity_boost(failed_nodes)

        while LS1:
            next_failures = set()
            
            for node in LS1:
                neighbors = list(self.G.successors(node)) if self.G.is_directed() else list(self.G.neighbors(node))
                sum_neighbors_capacity = sum(self.G.nodes[n]['capacity'] for n in neighbors if n not in failed_nodes)

                for neighbor in neighbors:
                    if neighbor not in failed_nodes:
                        if sum_neighbors_capacity == 0:
                            # If no available capacity, node fails immediately
                            next_failures.add(neighbor)
                        else:
                            redistributed_load = (
                                self.G.nodes[node]['load'] * (self.G.nodes[neighbor]['capacity'] / sum_neighbors_capacity)
                            )
                            self.G.nodes[neighbor]['load'] += redistributed_load

                            if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                                next_failures.add(neighbor)

            failed_nodes.update(next_failures)
            failed_nodes_list.extend(next_failures)
            LS2.update(LS1)
            LS1 = next_failures  # Update LS1 with the new set of failing nodes

            # Apply prevention mechanisms dynamically
            if use_prevention == "dynamic_load_redistribution":
                self.dynamic_load_redistribution(failed_nodes)
            elif use_prevention == "controlled_failure_isolation":
                self.controlled_failure_isolation(failed_nodes)
            elif use_prevention == "prevent_cascading_failure":
                self.prevent_cascading_failure(failed_nodes)

        NA = len(initial_failures)
        self.CF = NA / (len(failed_nodes) * self.N)
        I = len(failed_nodes) / self.N 

        return failed_nodes
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

    def rank_centrality(self, centrality_type='degree', length = None): 
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
