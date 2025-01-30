import numpy as np
import networkx as nx
import random

class PowerGridSimulator:
    def __init__(self, custom_network=None, min_impedance=0.005, max_impedance=0.05, load_threshold=1.0, verbose=False):
        self.min_impedance = min_impedance
        self.max_impedance = max_impedance
        self.load_threshold = load_threshold
        self.verbose = verbose
        
        if custom_network is not None:
            if not nx.is_connected(custom_network):
                raise ValueError("The provided network must be connected")
            self.G = custom_network
        else:
            raise ValueError("A network must be provided")
            
        self.num_nodes = self.G.number_of_nodes()
        self.impedances = self.assign_impedances()
        self.power_injections = self.generate_power_injections()

    def assign_impedances(self):
        impedances = np.zeros((self.num_nodes, self.num_nodes))
        for u, v in self.G.edges:
            impedance = np.random.uniform(self.min_impedance, self.max_impedance)
            impedances[u, v] = impedance
            impedances[v, u] = impedance
        return impedances

    def generate_power_injections(self):
        power_injections = np.zeros(self.num_nodes)
        power_injections[1:] = np.random.uniform(-1, 1, self.num_nodes - 1)
        power_injections[1:] -= np.sum(power_injections[1:]) / (self.num_nodes - 1)
        return power_injections
    
    def dc_power_flow(self, failed_nodes=None):
        if failed_nodes is None:
            failed_nodes = set()
        
        Y = np.zeros_like(self.impedances)
        np.fill_diagonal(Y, 0)
        
        for i in range(self.num_nodes):
            if i in failed_nodes:
                continue
            for j in range(self.num_nodes):
                if j in failed_nodes:
                    continue
                if self.impedances[i, j] != 0:
                    Y[i, j] = -1 / self.impedances[i, j]
                    Y[i, i] -= -1 / self.impedances[i, j]
        
        valid_nodes = [i for i in range(1, self.num_nodes) if i not in failed_nodes]
        Y_reduced = Y[np.ix_(valid_nodes, valid_nodes)]
        P_reduced = self.power_injections[valid_nodes]
        
        try:
            theta = np.linalg.solve(Y_reduced, P_reduced)
            theta_full = np.zeros(self.num_nodes)
            for i, node in enumerate(valid_nodes):
                theta_full[node] = theta[i]
            return theta_full, True
        except np.linalg.LinAlgError:
            return np.zeros(self.num_nodes), False
    
    def simulate_node_cascade(self, initial_failed_node):
        failed_nodes = {initial_failed_node}
        iterations = 0
        max_iterations = self.num_nodes
        cascade_history = [list(failed_nodes)]  # Track cascade progression
        
        while iterations < max_iterations:
            if self.verbose:
                print(f"\nIteration {iterations + 1}")
                print(f"Failed nodes: {failed_nodes}")
            
            voltage_angles, is_solvable = self.dc_power_flow(failed_nodes)
            if not is_solvable:
                if self.verbose:
                    print("System has become unstable - complete blackout")
                break
            
            node_loads = np.zeros(self.num_nodes)
            new_failures = set()
            
            for i in range(self.num_nodes):
                if i in failed_nodes:
                    continue
                load = 0
                for j in range(self.num_nodes):
                    if j in failed_nodes or self.impedances[i, j] == 0:
                        continue
                    flow = abs((voltage_angles[i] - voltage_angles[j]) / self.impedances[i, j])
                    load += flow
                
                node_loads[i] = load
                if load > self.load_threshold:
                    new_failures.add(i)
            
            if not new_failures:
                if self.verbose:
                    print("Cascade stopped - system stabilized")
                break
            
            failed_nodes.update(new_failures)
            cascade_history.append(list(failed_nodes))
            iterations += 1
        
        return failed_nodes, node_loads, cascade_history
    
    def run_simulation(self, initial_node=None):
        if initial_node is None:
            initial_node = random.randint(1, self.num_nodes - 1)
        print(f"\nInitiating cascade with failure of node {initial_node}")
        
        final_failed_nodes, final_loads, cascade_history = self.simulate_node_cascade(initial_node)
        
        print("\nFinal Results:")
        print(f"Total number of failed nodes: {len(final_failed_nodes)}")
        
        if self.verbose:
            print(f"Failed nodes: {sorted(final_failed_nodes)}")
            print("\nFinal loads on surviving nodes:")
            for i in range(self.num_nodes):
                if i not in final_failed_nodes:
                    print(f"Node {i}: {final_loads[i]:.3f}")
        
        surviving_nodes = set(range(self.num_nodes)) - final_failed_nodes
        original_connectivity = nx.average_node_connectivity(self.G)
        surviving_subgraph = self.G.subgraph(surviving_nodes)
        final_components = list(nx.connected_components(surviving_subgraph))
        
        print("\nNetwork Statistics:")
        print(f"Original network connectivity: {original_connectivity:.3f}")
        print(f"Number of separated components after cascade: {len(final_components)}")
        print(f"Sizes of separated components: {[len(c) for c in final_components]}")
        print(f"\nNumber of failed nodes: {len(final_failed_nodes)}")
        
        return {
            'failed_nodes': final_failed_nodes,
            'final_loads': final_loads,
            'cascade_history': cascade_history,
            'components': final_components
        }


G = nx.barabasi_albert_graph(50, 3)
simulator = PowerGridSimulator(custom_network=G, verbose=True)
results = simulator.run_simulation()