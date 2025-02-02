from iceland_network import *
import random

G = load_gen_into_networkx('iceland.m')

# Initialise nodes 
for node in G.nodes():
    generation = random.uniform(0, 100)
    load = random.uniform(0, 100)
    capacity = generation + random.uniform(0, 50)
    
    G.nodes[node]['generation'] = generation
    G.nodes[node]['load'] = load
    G.nodes[node]['capacity'] = capacity

def cascade_failure(G):
    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration {iteration}: {G.number_of_nodes()} nodes remaining")
        
        # Identify nodes that have failed
        failed_nodes = [node for node in list(G.nodes())
                        if G.nodes[node]['load'] > G.nodes[node]['capacity']]
        
        if not failed_nodes:
            print("No more failures")
            break
        
        print("Nodes failing in this iteration:", failed_nodes)
        
        # Loop over failed nodes
        for node in failed_nodes:
            load_to_distribute = G.nodes[node]['load']

            # List of neighbors before removal
            neighbors = list(G.neighbors(node))
            
            # Remove the failed node
            G.remove_node(node)
            if neighbors:
                extra_load = load_to_distribute / len(neighbors)
                for neighbor in neighbors:
                    if neighbor in G:
                        G.nodes[neighbor]['load'] += extra_load
    return G

G_final = cascade_failure(G)
