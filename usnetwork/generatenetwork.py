import networkx as nx
import pandas as pd

# Read the CSV file
csv_file = 'us_network.csv'
df = pd.read_csv(csv_file)

# Create an empty graph
G = nx.Graph()

# Add edges from the CSV file to the graph
# Assuming the CSV has columns 'source' and 'target' for the endpoints
for index, row in df.iterrows():
    G.add_edge(row["V1"], row["V2"])
nx.write_graphml(G, "us_network.graphml")
# # Print the edges of the graph
# print("Edges of the graph:")
# print(G.edges(data=True))

# # Optionally, visualize the graph
# import matplotlib.pyplot as plt

# pos = nx.spring_layout(G)  # Layout for the nodes
# nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=10, font_size=10)
# labels = nx.get_edge_attributes(G, 'weight')  # Show weights if available
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
