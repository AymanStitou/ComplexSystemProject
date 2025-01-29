import numpy as np
import networkx as nx
import random
from copy import deepcopy

def create_power_grid(num_nodes=20, connection_probability=0.2):
    while True:
        G = nx.erdos_renyi_graph(num_nodes, connection_probability)
        if nx.is_connected(G):
            break
    
    node_connections = nx.to_numpy_array(G)
    impedances = np.where(node_connections > 0, 
                         np.random.uniform(0.01, 0.1, (num_nodes, num_nodes)), 
                         0)
    impedances = (impedances + impedances.T) / 2
    
    return G, impedances

def dc_power_flow(impedances, power_injections, failed_nodes=None):
    if failed_nodes is None:
        failed_nodes = set()
    
    num_nodes = len(impedances)
    
    # Creating modified admittance matrix excluding failed nodes
    Y = np.zeros_like(impedances)
    np.fill_diagonal(Y, 0)
    
    for i in range(num_nodes):
        if i in failed_nodes:
            continue
        for j in range(num_nodes):
            if j in failed_nodes:
                continue
            if impedances[i,j] != 0:
                Y[i,j] = -1/impedances[i,j]
                Y[i,i] -= -1/impedances[i,j]
    
    # Removing slack bus and failed nodes
    valid_nodes = [i for i in range(1, num_nodes) if i not in failed_nodes]
    Y_reduced = Y[np.ix_(valid_nodes, valid_nodes)]
    P_reduced = power_injections[valid_nodes]
    
    try:
        # Solving for voltage angles
        theta = np.linalg.solve(Y_reduced, P_reduced)
        
        # Reconstruct full voltage angle vector
        theta_full = np.zeros(num_nodes)
        for i, node in enumerate(valid_nodes):
            theta_full[node] = theta[i]
        
        return theta_full, True
    except np.linalg.LinAlgError:
        # System became unsolvable (network disconected )
        return np.zeros(num_nodes), False

def simulate_node_cascade(G, impedances, power_injections, initial_failed_node, load_threshold=0.8):
    num_nodes = len(G)
    failed_nodes = {initial_failed_node}
    iterations = 0
    max_iterations = num_nodes  # Prevent infinite loops
    
    while iterations < max_iterations:
        print(f"\nIteration {iterations + 1}")
        print(f"Failed nodes: {failed_nodes}")
        
        # Calculate power flow with current failed nodes
        voltage_angles, is_solvable = dc_power_flow(impedances, power_injections, failed_nodes)
        
        if not is_solvable:
            print("System has become unstable - complete blackout")
            break
        
        # Calculating load on each node
        node_loads = np.zeros(num_nodes)
        new_failures = set()
        
        for i in range(num_nodes):
            if i in failed_nodes:
                continue
            
            # Calculating total power flow through node
            load = 0
            for j in range(num_nodes):
                if j in failed_nodes or impedances[i,j] == 0:
                    continue
                flow = abs((voltage_angles[i] - voltage_angles[j]) / impedances[i,j])
                load += flow
            
            node_loads[i] = load
            
            # Checking if node is overloaded
            if load > load_threshold:
                new_failures.add(i)
        
        if not new_failures:
            print("Cascade stopped - system stabilized")
            break
            
        print("Overloaded nodes that will fail:", new_failures)
        failed_nodes.update(new_failures)
        iterations += 1
    
    return failed_nodes, node_loads

# Create initial power grid
num_nodes = 20
G, impedances = create_power_grid(num_nodes)

# Generate random power injections
power_injections = np.zeros(num_nodes)
power_injections[1:] = np.random.uniform(-1, 1, num_nodes-1)
power_injections[1:] -= np.sum(power_injections[1:])/(num_nodes-1)

# Simulateing cascade starting with a random non-slack node failure
initial_failed_node = random.randint(1, num_nodes-1)
print(f"\nInitiating cascade with failure of node {initial_failed_node}")

final_failed_nodes, final_loads = simulate_node_cascade(G, impedances, power_injections, initial_failed_node)

print("\nFinal Results:")
print(f"Total number of failed nodes: {len(final_failed_nodes)}")
print(f"Failed nodes: {sorted(final_failed_nodes)}")
print("\nFinal loads on surviving nodes:")
for i in range(num_nodes):
    if i not in final_failed_nodes:
        print(f"Node {i}: {final_loads[i]:.3f}")

# Calculating and print network statistics
surviving_nodes = set(range(num_nodes)) - final_failed_nodes
original_connectivity = nx.average_node_connectivity(G)
surviving_subgraph = G.subgraph(surviving_nodes)
final_components = list(nx.connected_components(surviving_subgraph))

print(f"\nNetwork Statistics:")
print(f"Original network connectivity: {original_connectivity:.3f}")
print(f"Number of separated components after cascade: {len(final_components)}")
print(f"Sizes of separated components: {[len(c) for c in final_components]}")
num_failed_nodes = len(final_failed_nodes)

print(f"\nNumber of failed nodes: {num_failed_nodes}")