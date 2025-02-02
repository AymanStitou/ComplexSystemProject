import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from CascadingFailure import CascadingFailureSimulation

def load_network(filepath):
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
    return G

def initialize_simulation(G):
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()
    return simulation

def run_simulations(simulation, alpha_values, prevention_mechanisms, num_simulations):
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

def plot_results(results, alpha_values, prevention_mechanisms, num_simulations):
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
    plt.show()

def save_results_to_csv(results, alpha_values, num_simulations, filename):
    df = pd.DataFrame({"alpha": alpha_values})
    for mechanism in results:
        df[f"CF_{mechanism}"] = results[mechanism]["CF"]
        df[f"I_{mechanism}"] = results[mechanism]["I"]
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# filepath = "usnetwork/us_network.graphml"
# G = load_network(filepath)
# simulation = initialize_simulation(G)
    
# alpha_values = np.linspace(0, 0.6, 25)
# prevention_mechanisms = ["None", "localized_capacity_boost", "dynamic_load_redistribution", "controlled_failure_isolation", "prevent_cascading_failure"]
# num_simulations = 10
    
# results = run_simulations(simulation, alpha_values, prevention_mechanisms, num_simulations)
# plot_results(results, alpha_values, prevention_mechanisms, num_simulations)
# save_results_to_csv(results, alpha_values, num_simulations, "TemplateModel/cascading_failure_results.csv")

