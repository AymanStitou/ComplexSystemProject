import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random

from CascadingFailure_alg_2 import CascadingFailureSimulation
from mpl_toolkits.mplot3d import Axes3D

def run_simulation_single_pair(G, alpha, beta, initial_failures, centrality_type):
    # Create a new simulation instance with a fresh copy of G
    simulation = CascadingFailureSimulation(G.copy())
    
    # Calculate centrality measures
    simulation.calculate_centrality_measures()
    
    # Set initial load based on the specified centrality type
    simulation.calculate_initial_load(centrality_type=centrality_type)
    
    # Calculate node capacities
    simulation.calculate_capacity(alpha=alpha, beta=beta)
    
    # Simulate cascading failures
    failed_nodes_timestep, CF, I, failed_nodes_order = simulation.simulate_cascading_failure(initial_failures)
    
    return I  # Return the impact factor

def simulate_and_average_3D(
    G,
    alpha_values,
    beta_values,
    centrality_types,
    num_simulations=5,
    p_fail=0.01
):

    results_3D = {
        cent: np.zeros((len(beta_values), len(alpha_values))) 
        for cent in centrality_types
    }

    total_nodes = len(G)
    n_failures = max(1, int(total_nodes * p_fail))

    for j, b in enumerate(beta_values):
        for i, a in enumerate(alpha_values):
            for cent in centrality_types:
                frac_acc = 0.0
                for _ in range(num_simulations):
                    initial_failures = random.sample(list(G.nodes()), n_failures)
                    
                    # Run the simulation and accumulate the impact factor
                    I = run_simulation_single_pair(G, a, b, initial_failures, cent)
                    frac_acc += I
                # Store the average impact factor
                results_3D[cent][j, i] = frac_acc / num_simulations
    return results_3D

def save_results_3D_to_csv(results_3D, alpha_vals, beta_vals, filename):
    rows = []
    for cent, matrix in results_3D.items():
        for j, b in enumerate(beta_vals):
            for i, a in enumerate(alpha_vals):
                rows.append({
                    "alpha": a,
                    "beta": b,
                    "I": matrix[j, i],
                    "centrality": cent
                })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")

if __name__ == "__main__":
    # Load the graph from a GraphML file
    G3 = nx.read_graphml("toy_network_undirected.graphml")
    
    # Optionally, relabel nodes to integers if necessary
    mapping3 = {node: int(node) for node in G3.nodes()}
    G3 = nx.relabel_nodes(G3, mapping3)
    
    # Define parameter ranges
    alpha_vals = np.linspace(0, 1.2, 10)
    beta_vals = np.linspace(0, 2.0, 10)
    centrality_types = ["degree", "betweenness", "closeness"]
    
    # Run simulations
    results_us_3D = simulate_and_average_3D(
        G3, 
        alpha_vals, 
        beta_vals, 
        centrality_types, 
        num_simulations=25, 
        p_fail=0.01
    )
    
    # Plotting the results
    fig = plt.figure(figsize=(18, 6))
    A, B = np.meshgrid(alpha_vals, beta_vals) 
    
    for idx, cent in enumerate(centrality_types, start=1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        Z = results_us_3D[cent]  

        surf = ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_title(f"Centrality: {cent.capitalize()}")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\beta$")
        ax.set_zlabel("Fraction Failed (I)")
        ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig("us_network_3D_results.png")
    plt.show()
    
    # Save the results to a CSV file
    save_results_3D_to_csv(results_us_3D, alpha_vals, beta_vals, "us_network_3D_results.csv")
