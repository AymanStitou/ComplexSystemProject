from CascadingFailure import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

# Load network
G = nx.read_graphml("usnetwork/us_network.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)
total_nodes = len(G.nodes())
print(total_nodes)
# Initialize simulation
simulation = CascadingFailureSimulation(G)
simulation.calculate_centrality_measures()

# Define alpha values and prevention mechanisms
alpha_values = np.linspace(0.2, 0.6, 25)
prevention_mechanisms = ["None", "localized_capacity_boost", "dynamic_load_redistribution", "controlled_failure_isolation", "prevent_cascading_failure"]
num_simulations = 20  # Number of runs to average over

# Storage for results (accumulate values across simulations)
results = {mechanism: {"CF": np.zeros(len(alpha_values)), "I": np.zeros(len(alpha_values))} for mechanism in prevention_mechanisms}

# Run multiple simulations
for sim in range(num_simulations):
    print(f"Running simulation {sim+1}/{num_simulations}")

    # Generate new random failures for each run
    initial_failures = [random.randint(1, total_nodes) for _ in range(int(total_nodes/100))]

    for mechanism in prevention_mechanisms:
        print(f"Simulating with prevention mechanism: {mechanism}")

        for idx, alpha in enumerate(alpha_values):
            simulation.calculate_initial_load(centrality_type='degree')
            simulation.calculate_capacity(alpha=alpha, beta=1)

            failed_nodes, CF, I, failed_nodes_list = simulation.simulate_cascading_failure(
                initial_failures, use_prevention=mechanism
            )

            # Accumulate results
            results[mechanism]["CF"][idx] += CF
            results[mechanism]["I"][idx] += I

# Compute averages over 50 simulations
for mechanism in prevention_mechanisms:
    results[mechanism]["CF"] /= num_simulations
    results[mechanism]["I"] /= num_simulations

# Define marker styles for different prevention mechanisms
markers = ["o", "s", "D", "^", "x"]
line_styles = ["-", "--", "-.", ":", "-"]

# Plot CF vs alpha (Averaged)
plt.figure(figsize=(10, 6))
for i, mechanism in enumerate(prevention_mechanisms):
    plt.plot(alpha_values, results[mechanism]["CF"], 
             marker=markers[i], linestyle=line_styles[i], 
             label=mechanism, markersize=6)
plt.xlabel("Value of alpha")
plt.ylabel("Average CF")
plt.title(f"Cascading Failure Robustness (CF) vs Alpha (Averaged over {num_simulations} Simulations)")
plt.legend()
plt.grid()
plt.show()

# Plot I vs alpha (Averaged)
plt.figure(figsize=(10, 6))
for i, mechanism in enumerate(prevention_mechanisms):
    plt.plot(alpha_values, results[mechanism]["I"], 
             marker=markers[i], linestyle=line_styles[i], 
             label=mechanism, markersize=6)
plt.xlabel("Value of alpha")
plt.ylabel("Average I")
plt.title(f"Fraction of Failed Nodes (I) vs Alpha (Averaged over {num_simulations} Simulations)")
plt.legend()
plt.grid()
plt.show()

# Print final results for averaged CF and I
print("Final Averaged Results:")
for mechanism in prevention_mechanisms:
    print(f"{mechanism} -> Avg CF: {results[mechanism]['CF'][-1]:.4f}, Avg I: {results[mechanism]['I'][-1]:.4f}")

# Save results to CSV file
csv_filename = f"TemplateModel/cascading_failure_results_{num_simulations}_runs.csv"
df = pd.DataFrame({"alpha": alpha_values})

for mechanism in prevention_mechanisms:
    df[f"CF_{mechanism}"] = results[mechanism]["CF"]
    df[f"I_{mechanism}"] = results[mechanism]["I"]

df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")