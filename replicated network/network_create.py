# create own network
import networkx as nx
import matplotlib.pyplot as plt

nodes = range(25)
edges = [
    (1, 0), (2, 1), (3, 1), (4, 1),(6,1), (5, 2), (6, 2), (7, 3), (8, 5), (9, 6), 
    (9, 7), (8, 9), (10, 9), (11, 10), (12, 11), (13, 11), (14, 12), 
    (15, 13), (16, 14), (16, 15), (18, 16), (17, 13), (18, 17), 
    (20, 19), (21, 19), (22, 21), (22, 20),(23, 21),(24,22),(24,23)
    ]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
nx.write_graphml(G, "custom_network.graphml")

plt.figure(figsize=(10, 6))
nx.draw(
    G, with_labels=True, node_size=300, node_color="blue", font_size=10, arrows=True
    )
plt.title("Custom Directed Network")
plt.show()