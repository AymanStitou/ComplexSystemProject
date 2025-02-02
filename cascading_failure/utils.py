import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from .CascadingFailure import CascadingFailureSimulation
from mpl_toolkits.mplot3d import Axes3D

def load_network(filepath):
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
    return G

def load_attack_results_from_csv(filename):
    """
    Load simulation results from a CSV file.

    """
    df = pd.read_csv(filename)
    
    alpha = df["Alpha"].tolist()
    results = df.drop(columns=["Alpha"]).to_dict(orient="list")
    
    return alpha, results

def initialize_simulation(G):
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()
    return simulation

def simulation_capacity(initial_failures, centrality, simulation, capacity_list): 
    sum_centrality = simulation.calculate_centrality_measures()
    I_list = []
    for c in capacity_list: 
        simulation.calculate_initial_load(centrality_type=centrality, sum_centrality=sum_centrality)
        simulation.calculate_capacity(total_capacity=c)
        _, _, I, _ = simulation.simulate_cascading_failure(initial_failures)
        I_list.append(I)
    return I_list

def run_prevention_mechanism_simulations(simulation, alpha_values, prevention_mechanisms, num_simulations):
    total_nodes = len(simulation.G.nodes())
    results = {mechanism: {"CF": np.zeros(len(alpha_values)), "I": np.zeros(len(alpha_values)), "total_capacity": np.zeros(len(alpha_values))} for mechanism in prevention_mechanisms}
    
    for sim in range(num_simulations):
        print(f"Running simulation {sim+1}/{num_simulations}")
        initial_failures = [random.randint(1, total_nodes) for _ in range(int(total_nodes / 100))]
        
        for mechanism in prevention_mechanisms:
            print(f"Simulating with prevention mechanism: {mechanism}")
            
            for idx, alpha in enumerate(alpha_values):
                simulation.calculate_initial_load(centrality_type='degree')
                simulation.calculate_capacity(alpha=alpha, beta=1.2)
                
                _, CF, I, _ = simulation.simulate_cascading_failure(initial_failures, use_prevention=mechanism)
                results[mechanism]["CF"][idx] += CF
                results[mechanism]["I"][idx] += I
                results[mechanism]["total_capacity"][idx] += simulation.return_total_capacity()
    
    for mechanism in prevention_mechanisms:
        results[mechanism]["CF"] /= num_simulations
        results[mechanism]["I"] /= num_simulations
        results[mechanism]["total_capacity"] /= num_simulations
    
    return results

def run_target_attack_simulations(initial_failures, centrality_type, simulation, alpha=0.2, beta=1, alpha_list=None, beta_list=None, use_prevention=False):
    """
    Run the cascading failure simulation for a specific network and centrality measure.
    It returns a list of the number of failed nodes for each alpha value.
    """
    I_list = []

    if alpha_list is not None: 
        for a in alpha_list:
            simulation.calculate_initial_load(centrality_type=centrality_type)
            simulation.calculate_capacity(alpha=a, beta=beta)  
            _, _, I, _ = simulation.simulate_cascading_failure(initial_failures)
            I_list.append(I)
    
    if beta_list is not None: 
        for b in beta_list: 
            simulation.calculate_initial_load(centrality_type=centrality_type)
            simulation.calculate_capacity(alpha=alpha, beta=b) 
            _, _, I, _ = simulation.simulate_cascading_failure(initial_failures)
            I_list.append(I)

    return I_list

def run_simulation_single_pair(G, alpha, beta, initial_failures, centrality_type, simulation):
    simulation.calculate_initial_load(centrality_type=centrality_type)
    simulation.calculate_capacity(alpha=alpha, beta=beta)
    failed_nodes = simulation.simulate_cascading_failure(initial_failures)
    return len(failed_nodes) / len(G) 

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

def simulate_and_average_capacity(G, centrality_types, capacity_list, num_simulations=25, target_attack=False):
    results = {centrality: [] for centrality in centrality_types}
    total_nodes = len(G.nodes)
    num_failures = max(1, int(total_nodes * 0.01))
    simulation = CascadingFailureSimulation(G)
    if target_attack: 
        for centrality in centrality_types:
            initial_failures = simulation.rank_centrality(centrality, num_failures)
            I = simulation_capacity(initial_failures, centrality, simulation, capacity_list)
            results[centrality] = I
            print(fr"Finish simulation of the centrality type: {centrality}")
            
        return results
    else: 
        for _ in range(num_simulations):
            initial_failures = random.sample(range(1,total_nodes-1), num_failures)
            for centrality in centrality_types:
                I = simulation_capacity(initial_failures, centrality, simulation, capacity_list)
                results[centrality].append(I)
                print(fr"Finish simulation of the centrality type: {centrality}")

        # Compute mean I_list for each centrality type across simulations
        mean_results = {centrality: np.mean(results[centrality], axis=0) for centrality in centrality_types}
        
        return mean_results

def simulate_and_average(G, alpha, centrality_types, target_attack = False, num_simulations=30, beta = 1.2): # run each network for 25 times
    """
    Simulate the cascading failure multiple times and calculate the mean fraction of failed nodes for each centrality type.
    Return a dictionary with centrality measures as keys and mean I_list as values.
    """
    results = {centrality: [] for centrality in centrality_types}
    total_nodes = len(G.nodes)
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()
    num_failures = max(1, int(total_nodes * 0.01))
    if target_attack: 
        for centrality in centrality_types:
            initial_failures = simulation.rank_centrality(centrality, num_failures) 
            I = run_simulation(G, alpha, initial_failures, centrality, simulation, beta = beta)
            results[centrality] = (I, np.zeros_like(I))
            return results
    else:
        for i in range(num_simulations):
            initial_failures = random.sample(range(1,total_nodes-1), num_failures)
            
            for centrality in centrality_types:
                I = run_simulation(G, alpha, initial_failures, centrality, simulation, beta = beta)
                results[centrality].append(I)
    
        # Compute mean I_list for each centrality type across simulations
        mean_results = {centrality: np.mean(results[centrality], axis=0) for centrality in centrality_types}
        std_results = {centrality: np.std(results[centrality], axis=0, ddof=1) for centrality in centrality_types}
        return {centrality: (mean_results[centrality], std_results[centrality]) for centrality in centrality_types}



def simulate_and_average_3D(G, alpha_values, beta_values, centrality_types, num_simulations=5, p_fail=0.01):
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

def plot_prevention_mechanism_results(results, alpha_values, prevention_mechanisms, num_simulations, saveplot):
    markers = ["o", "s", "D", "^", "x"]
    line_styles = ["-", "--", "-.", ":", "-"]
    
    plt.figure(figsize=(10, 6))
    for i, mechanism in enumerate(prevention_mechanisms):
        plt.plot(results[mechanism]["total_capacity"], results[mechanism]["CF"], 
                 marker=markers[i], linestyle=line_styles[i], 
                 label=mechanism, markersize=6)
    plt.xlabel("Average total capacity")
    plt.ylabel("Average CF")
    plt.title(f"Cascading Failure Robustness (CF) vs Avg tot. capacity (Averaged over {num_simulations} Simulations)")
    plt.legend()
    plt.grid()

    if saveplot:
        plt.savefig('results/plots/prevention_mechanism_CF.png')
    plt.show()

    
    plt.figure(figsize=(10, 6))
    for i, mechanism in enumerate(prevention_mechanisms):
        plt.plot(results[mechanism]["total_capacity"], results[mechanism]["I"], 
                 marker=markers[i], linestyle=line_styles[i], 
                 label=mechanism, markersize=6)
    plt.xlabel("Average total capacity")
    plt.ylabel("Average I")
    plt.title(f"Fraction of Failed Nodes (I) vs Avg tot. capacity (Averaged over {num_simulations} Simulations)")
    plt.legend()
    plt.grid()
    if saveplot:
        plt.savefig('results/plots/prevention_mechanism_CF.png')
    plt.show()

def save_prevention_results_to_csv(results, alpha_values, num_simulations, filename):
    df = pd.DataFrame({"alpha": alpha_values})
    for mechanism in results:
        df[f"CF_{mechanism}"] = results[mechanism]["CF"]
        df[f"I_{mechanism}"] = results[mechanism]["I"]
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot_attack_graphs(results, alpha=0.2, beta=1, alpha_list=None, beta_list=None, capacity_list=None, network_type=None, file_name=None): 
    # plot the figures for the three network
    plt.figure(figsize=(10, 6))
    for centrality, mean_result in results.items():
        if alpha_list is not None:
            plt.plot(alpha_list, mean_result, label=f"{centrality.capitalize()} Centrality", marker='o')
            plt.title(fr"Mean Fraction of Failed Nodes vs. $\alpha$ ({network_type}), with $\beta$={beta}")
            plt.xlabel(fr"$\alpha$")
        elif beta_list is not None:
            plt.plot(beta_list, mean_result, label=f"{centrality.capitalize()} Centrality", marker='o')
            plt.title(fr"Mean Fraction of Failed Nodes vs. $\beta$ ({network_type}), with $\alpha$={alpha}")
            plt.xlabel(fr"$\beta$")
        elif capacity_list is not None: 
            plt.plot(capacity_list, mean_result, label=f"{centrality.capitalize()} Centrality", marker='o')
            plt.title(fr"Mean Fraction of Failed Nodes vs. Total Capacity ({network_type})")
            plt.xlabel("Total Capacity")
        else:
            raise ValueError("No input of varying variables (alpha/beta)")
    plt.ylabel("Mean Fraction of Failed Nodes (I)")
    plt.legend()
    plt.grid()

    if file_name is not None: 
        if alpha_list is not None: 
            plt.savefig(fr'results/plots/{file_name}_beta_{beta}.png') 
        elif beta_list is not None: 
            plt.savefig(fr'results/plots/{file_name}_alpha_{alpha}.png') 
        elif capacity_list is not None:
            plt.savefig(fr'results/plots/{file_name}.png') 
        else: 
            raise ValueError("No input of varying variables (alpha/beta)")
    else: 
        plt.show()

def plot_3D_results_from_csv(filename):

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


def save_attack_results_to_csv(results, filename, alpha_list=None, beta_list=None, capacity_list=None):
    """
    Save simulation results to a CSV file.

    """
    df = pd.DataFrame(results)
    if alpha_list is not None: 
        df.insert(0, "Alpha", alpha_list)
    elif beta_list is not None: 
        df.insert(0, "Beta", beta_list)
    elif capacity_list is not None: 
        df.insert(0, "Total_Capacity", capacity_list)
    else: 
        raise ValueError("No input of varying variables (alpha/beta)")
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

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


