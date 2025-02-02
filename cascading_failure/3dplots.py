import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random

from CascadingFailure import CascadingFailureSimulation
from mpl_toolkits.mplot3d import Axes3D

def run_simulation_single_pair(G, alpha, beta, initial_failures, centrality_type, simulation):
    simulation.calculate_initial_load(centrality_type=centrality_type)
    simulation.calculate_capacity(alpha=alpha, beta=beta)
    failed_nodes = simulation.simulate_cascading_failure(initial_failures)
    return len(failed_nodes) / len(G) 

def simulate_and_average_3D(
    G,
    alpha_values,
    beta_values,
    centrality_types,
    num_simulations=5,
    p_fail=0.01
):
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()

    results_3D = {
        cent: np.zeros((len(beta_values), len(alpha_values))) 
        for cent in centrality_types
    }

    total_nodes = len(G)
    n_failures = max(1, int(total_nodes * p_fail))

    for j, b in enumerate(beta_values):
        print(f"Currently at {j+1}/{len(beta_values)} for beta value {b}")
        for i, a in enumerate(alpha_values):
            
            print(f"Currently at {i+1}/{len(alpha_values)} for alpha value {a}")
            for cent in centrality_types:
                frac_acc = 0.0
                for _ in range(num_simulations):
                    initial_failures = random.sample(list(G.nodes()), n_failures)
                    frac_acc += run_simulation_single_pair(G, a, b, initial_failures, cent, simulation)
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
    print(f"results saved to: {filename}")



def plot_results_from_csv(filename):

    df = pd.read_csv(filename)

    # Get unique centrality types
    centrality_types = df['centrality'].unique()
    alpha_vals = np.sort(df['alpha'].unique())
    beta_vals = np.sort(df['beta'].unique())
    A, B = np.meshgrid(alpha_vals, beta_vals)
    num_centralities = len(centrality_types)
    fig = plt.figure(figsize=(6 * num_centralities, 6))  

    for idx, cent in enumerate(centrality_types, start=1):
        ax = fig.add_subplot(1, num_centralities, idx, projection='3d')
        pivot_df = df[df['centrality'] == cent].pivot(index='beta', columns='alpha', values='I')

        Z = pivot_df.values 
        surf = ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title(f"Centrality: {cent.capitalize()}", fontsize=14)
        ax.set_xlabel(r"$\alpha$", fontsize=12)
        ax.set_ylabel(r"$\beta$", fontsize=12)
        ax.set_zlabel("Fraction Failed (I)", fontsize=12)
        ax.view_init(elev=25, azim=60)  

    plt.tight_layout()
    plt.show()
