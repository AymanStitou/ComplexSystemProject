import networkx as nx
import matplotlib.pyplot as plt
import random

class CascadingFailureSimulation:
    def __init__(self, num_nodes=25, edges_per_node=2):
        # self.G = nx.barabasi_albert_graph(num_nodes, edges_per_node)
        self.G = nx.erdos_renyi_graph(25,0.2)
        for node in self.G.nodes:
            self.G.nodes[node]['load'] = 0
            self.G.nodes[node]['capacity'] = 0
            self.G.nodes[node]['degree_centrality'] = 0
            self.G.nodes[node]['betweenness_centrality'] = 0
            self.G.nodes[node]['closeness_centrality'] = 0

    def calculate_centrality_measures(self):
        degree_centrality = nx.degree_centrality(self.G)
        betweenness_centrality = nx.betweenness_centrality(self.G, normalized=True)
        closeness_centrality = nx.closeness_centrality(self.G)

        for node in self.G.nodes:
            self.G.nodes[node]['degree_centrality'] = degree_centrality[node]
            self.G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]
            self.G.nodes[node]['closeness_centrality'] = closeness_centrality[node]

    def calculate_initial_load(self, centrality_type='degree'):
        for node in self.G.nodes:
            if centrality_type == 'degree':
                self.G.nodes[node]['load'] = self.G.nodes[node]['degree_centrality'] * 10
            elif centrality_type == 'betweenness':
                self.G.nodes[node]['load'] = self.G.nodes[node]['betweenness_centrality'] * 10
            elif centrality_type == 'closeness':
                self.G.nodes[node]['load'] = self.G.nodes[node]['closeness_centrality'] * 10

    def calculate_capacity(self, alpha=0.2, beta=1.5):
        for node in self.G.nodes:
            load = self.G.nodes[node]['load']
            self.G.nodes[node]['capacity'] = (1 + alpha) * (load ** beta)

    def simulate_cascading_failure(self, initial_failures):
        failed_nodes = set(initial_failures)
        queue = list(initial_failures)

        # I think something goes wrong here
        while queue:
            node = queue.pop(0)
            neighbors = list(self.G.neighbors(node))
            # print(list(self.G.neighbors(node)))
            sum_neighbours = sum([self.G.nodes[neighbor]['load'] for neighbor in neighbors])
            # print(sum_neighbours)
            if neighbors:
                # load_share = self.G.nodes[node]['load'] / len(neighbors)
                for neighbor in neighbors:
                    if neighbor not in failed_nodes:
                        self.G.nodes[neighbor]['load'] += self.G.nodes[node]['load'] * self.G.nodes[neighbor]['load']/sum_neighbours
                        print(self.G.nodes[neighbor]['load'],self.G.nodes[neighbor]['capacity'])
                        if self.G.nodes[neighbor]['load'] > self.G.nodes[neighbor]['capacity']:
                            failed_nodes.add(neighbor)
                            queue.append(neighbor)

        return failed_nodes

    def visualize_network(self, failed_nodes):
        pos = nx.spring_layout(self.G)
        node_colors = ['red' if node in failed_nodes else 'green' for node in self.G.nodes]
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=500)
        plt.show()

    def print_centrality_measures(self):
        print("Node centrality measures:")
        for node in self.G.nodes:
            print(f"Node {node}: Degree={self.G.nodes[node]['degree_centrality']:.2f}, "
                  f"Betweenness={self.G.nodes[node]['betweenness_centrality']:.2f}, "
                  f"Closeness={self.G.nodes[node]['closeness_centrality']:.2f}")