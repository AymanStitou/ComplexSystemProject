import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

class PowerNetwork:
    def __init__(self):
        self.network = nx.Graph()
        self.reference_bus = None
        self.bus_data = {}
        
    def add_branch(self, from_bus, to_bus, impedance):
        """
        Add a branch between two buses
        
        Parameters:
        from_bus: source bus ID
        to_bus: destination bus ID
        impedance: complex impedance of the branch
        """
        if from_bus not in self.bus_data or to_bus not in self.bus_data:
            raise ValueError("Both buses must exist in the network")
        
        self.network.add_edge(from_bus, to_bus, impedance=impedance)
        
    def generate_er_network(self, n_buses, probability, base_voltage=1.0):
        """
        Generate an Erdős-Rényi random network with specified bus types
        """
        # Clear existing network
        self.network.clear()
        self.bus_data.clear()
        self.reference_bus = None
        
        # Create ER network
        er_network = nx.erdos_renyi_graph(n_buses, probability)
        
        # Calculate number of buses of each type
        n_gen = int(0.2 * n_buses)  # 20% generators
        n_load = int(0.5 * n_buses)  # 50% loads
        n_interconn = int(0.2 * n_buses)  # 20% interconnection
        
        # Create list of bus types
        bus_types = (['generator'] * n_gen + 
                    ['load'] * n_load + 
                    ['interconnection'] * n_interconn +
                    ['other'] * (n_buses - n_gen - n_load - n_interconn - 1))
        
        # Add reference bus first
        self.add_bus(0, voltage=base_voltage, is_reference=True, bus_type='reference')
        
        # Shuffle and assign remaining bus types
        random.shuffle(bus_types)
        for i in range(1, n_buses):
            bus_type = bus_types[i-1]
            self.add_bus(i, voltage=base_voltage, is_reference=False, bus_type=bus_type)
            
            # Assign typical power values based on bus type
            if bus_type == 'generator':
                p_gen = random.uniform(50, 200)  # Random generation between 50-200 MW
                self.set_bus_power(i, p_gen=p_gen, q_gen=0.2*p_gen)  # Assuming 0.2 power factor
            elif bus_type == 'load':
                p_load = random.uniform(20, 100)  # Random load between 20-100 MW
                self.set_bus_power(i, p_load=p_load, q_load=0.3*p_load)  # Assuming 0.3 power factor
        
        # Add branches from ER network with random impedances
        for edge in er_network.edges():
            r = random.uniform(0.01, 0.05)  # resistance
            x = random.uniform(0.05, 0.15)  # reactance
            self.add_branch(edge[0], edge[1], r + x*1j)
    
    def add_bus(self, bus_id, voltage=1.0, is_reference=False, bus_type='other'):
        """
        Add a bus to the network with specified type
        """
        self.network.add_node(bus_id)
        self.bus_data[bus_id] = {
            'voltage': voltage,
            'type': 'reference' if is_reference else bus_type,
            'p_gen': 0.0,
            'q_gen': 0.0,
            'p_load': 0.0,
            'q_load': 0.0
        }
        
        if is_reference:
            if self.reference_bus is not None:
                raise ValueError("Network already has a reference bus")
            self.reference_bus = bus_id
    
    def set_bus_power(self, bus_id, p_gen=0.0, q_gen=0.0, p_load=0.0, q_load=0.0):
        """Set power generation and load at a bus"""
        if bus_id not in self.bus_data:
            raise ValueError(f"Bus {bus_id} does not exist in the network")
        
        self.bus_data[bus_id].update({
            'p_gen': p_gen,
            'q_gen': q_gen,
            'p_load': p_load,
            'q_load': q_load
        })
    
    def visualize(self):
        """Create a visualization of the network with color-coded bus types"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.network)
        
        # Define colors for different bus types
        color_map = {
            'reference': 'red',
            'generator': 'green',
            'load': 'blue',
            'interconnection': 'yellow',
            'other': 'gray'
        }
        
        # Draw nodes with different colors based on type
        for bus_type in color_map:
            nodes = [node for node in self.network.nodes() 
                    if self.bus_data[node]['type'] == bus_type]
            if nodes:
                nx.draw_networkx_nodes(self.network, pos,
                                     nodelist=nodes,
                                     node_color=color_map[bus_type],
                                     node_size=500,
                                     label=bus_type)
        
        # Draw edges
        nx.draw_networkx_edges(self.network, pos)
        
        # Add labels
        labels = {node: f"Bus {node}\n{self.bus_data[node]['type'][:3]}" 
                 for node in self.network.nodes()}
        nx.draw_networkx_labels(self.network, pos, labels)
        
        plt.title("Power System Network")
        plt.legend()
        plt.axis('off')
        
        # Add bus type statistics
        bus_types = [data['type'] for data in self.bus_data.values()]
        stats = f"Bus Statistics:\n"
        for bus_type in set(bus_types):
            count = bus_types.count(bus_type)
            percentage = (count / len(bus_types)) * 100
            stats += f"{bus_type}: {count} ({percentage:.1f}%)\n"
        
        plt.figtext(1.02, 0.5, stats, fontsize=10)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a new network
    power_sys = PowerNetwork()
    
    # Generate ER network with 20 buses and 0.3 probability of connection
    power_sys.generate_er_network(n_buses=40, probability=0.2)
    
    # Visualize the network
    power_sys.visualize()