from CascadingFailure import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt 
import random
import pandas as pd
import numpy as np

def run_simulation(G, alpha, initial_failures, centrality_type, simulation, beta=1):
    """
    Run the cascading failure simulation for a specific network and centrality measure.
    It returns a list of the number of failed nodes for each alpha value.
    """
    n_failed_nodes = []
    I_list = []

    for a in alpha:
        simulation.calculate_initial_load(centrality_type=centrality_type)
        simulation.calculate_capacity(alpha=a, beta=beta)  # Fix beta to 1
        failed_nodes = simulation.simulate_cascading_failure(initial_failures)
        n_failed_nodes.append(len(failed_nodes))
        I_list.append(len(failed_nodes)/len(G))

    return I_list

def simulate_and_average(G, alpha, centrality_types, num_simulations=25, beta=1):
    """
    Simulate the cascading failure multiple times and calculate the mean fraction of failed nodes for each centrality type.
    Return a dictionary with centrality measures as keys and mean I_list as values.
    """
    results = {centrality: [] for centrality in centrality_types}
    total_nodes = len(G.nodes)
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()

    for i in range(num_simulations):
        num_failures = max(1, int(total_nodes * 0.01))  # 1% random failures
        initial_failures = random.sample(range(1,total_nodes-1), num_failures)
        
        for centrality in centrality_types:
            I = run_simulation(G, alpha, initial_failures, centrality, simulation, beta=beta)
            results[centrality].append(I)
    
    # Compute mean I_list for each centrality type across simulations
    mean_results = {centrality: np.mean(results[centrality], axis=0) for centrality in centrality_types}
    return mean_results


def load_results_from_csv(filename):
    """
    Load simulation results from a CSV file.

    """
    
    df = pd.read_csv(filename)
    
    alpha = df["Alpha"].tolist()
    results = df.drop(columns=["Alpha"]).to_dict(orient="list")
    
    return alpha, results


def save_results_to_csv(results, alpha, filename):
    """
    Save simulation results to a CSV file.

    """
    df = pd.DataFrame(results)
    df.insert(0, "Alpha", alpha)
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# load the toy network
G = nx.read_graphml("toy_network_undirected.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)

# load the iceland network
G2 = nx.read_graphml("iceland.graphml")
mapping2 = {node: int(node) for node in G2.nodes()}
G2 = nx.relabel_nodes(G2, mapping2)

# load the US network
G3 = nx.read_graphml("us_network.graphml")
mapping3 = {node: int(node) for node in G3.nodes()}
G3 = nx.relabel_nodes(G3, mapping3)


# initialize parameters and empty lists
alpha = np.linspace(0,1.2,10)
centrality_types = ["degree", "betweenness", "closeness"]

# start simulation of the toy network
results_toy = simulate_and_average(G, alpha, centrality_types)
print(f"The result for the toy network is: {results_toy}")

# start simulation of the iceland network
results_ice = simulate_and_average(G2, alpha, centrality_types)
print(f"The result for the iceland network is: {results_ice}")

# start simulation of the US network
results_us = simulate_and_average(G3, alpha, centrality_types)
print(f"The result for the US network is: {results_us}")


# plot the figures for the three network
plt.figure(figsize=(10, 6))
for centrality, mean_result in results_toy.items():
    plt.plot(alpha, mean_result, label=f"{centrality.capitalize()} Centrality", marker='o')
plt.title("Mean Fraction of Failed Nodes vs. Alpha (Toy Network)")
plt.xlabel("Alpha")
plt.ylabel("Mean Fraction of Failed Nodes (I)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for centrality, mean_result in results_ice.items():
    plt.plot(alpha, mean_result, label=f"{centrality.capitalize()} Centrality", marker='o')
plt.title("Mean Fraction of Failed Nodes vs. Alpha (Iceland Network)")
plt.xlabel("Alpha")
plt.ylabel("Mean Fraction of Failed Nodes (I)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for centrality, mean_result in results_us.items():
    plt.plot(alpha, mean_result, label=f"{centrality.capitalize()} Centrality", marker='o')
plt.title("Mean Fraction of Failed Nodes vs. Alpha (US Network)")
plt.xlabel("Alpha")
plt.ylabel("Mean Fraction of Failed Nodes (I)")
plt.legend()
plt.grid()
plt.show()

# save simulation results to csv file:
save_results_to_csv(results_toy, alpha, "toy_network_results.csv")
save_results_to_csv(results_ice, alpha, "iceland_network_results.csv")
save_results_to_csv(results_us, alpha, "US_network_results.csv")

