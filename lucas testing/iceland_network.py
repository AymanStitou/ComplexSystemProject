import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def load_gen_into_networkx(filename):
    """
    Loads only the mpc.gen block from a MATPOWER (.m) file named `filename`,
    parses the first two columns as edges, and returns a NetworkX graph.
    """
    G = nx.Graph()  
    reading_gen_block = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Look for the start of the mpc.gen block
            if line.startswith('mpc.branch = ['):
                reading_gen_block = True
                continue
            
            # Break if at mpc.gen its end
            if reading_gen_block:
                if line.startswith('];'):
                    reading_gen_block = False
                    break  
                
                # Split data
                line = line.replace(';', '')
                columns = line.split()

                # Only first two columns                
                if len(columns) < 2:
                    continue
                
                from_bus = int(float(columns[0]))
                to_bus   = int(float(columns[1]))
                
                # Add edge 
                G.add_edge(from_bus, to_bus)
                
    return G

G = load_gen_into_networkx('iceland.m')
print("Nodes:", len(G.nodes()))
print("Edges:", len(G.edges()))

graph = load_gen_into_networkSx('iceland.m')

pos = nx.spring_layout(graph)  
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')

plt.title("Network from mpc.gen data")
plt.show()
