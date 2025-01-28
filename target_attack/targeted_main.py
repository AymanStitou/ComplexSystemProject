from CascadingFailure import CascadingFailureSimulation
import networkx as nx
import matplotlib.pyplot as plt 
import random
import pandas as pd
import numpy as np

def run_simulation(initial_failures, centrality_type, simulation, alpha=0.2, beta=1, alpha_list=None, beta_list=None):
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

def simulate_and_average(G, centrality_types, num_simulations=25, target_attack=False, alpha=0.2, beta=1, alpha_list=None, beta_list=None):
    """
    Simulate the cascading failure multiple times and calculate the mean fraction of failed nodes for each centrality type.
    Return a dictionary with centrality measures as keys and mean I_list as values.
    """
    results = {centrality: [] for centrality in centrality_types}
    total_nodes = len(G.nodes)
    num_failures = max(1, int(total_nodes * 0.01)) # 1% random failures
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()

    if target_attack: 
        for centrality in centrality_types:
            initial_failures = simulation.rank_centrality(centrality, num_failures)
            if alpha_list is not None: 
                I = run_simulation(initial_failures, centrality, simulation, alpha_list=alpha_list, beta=beta)
            elif beta_list is not None: 
                I = run_simulation(initial_failures, centrality, simulation, beta_list=beta_list, alpha=alpha)
            else: 
                raise ValueError
            results[centrality] = I

        return results
    
    else: 
        for _ in range(num_simulations, attacked_type='random'):
            initial_failures = random.sample(range(1,total_nodes-1), num_failures)
            for centrality in centrality_types:
                if alpha_list is not None: 
                    I = run_simulation(initial_failures, centrality, simulation, alpha_list=alpha_list, beta=beta)
                elif beta_list is not None: 
                    I = run_simulation(initial_failures, centrality, simulation, beta_list=beta_list, alpha=alpha)
                else: 
                    raise ValueError
                results[centrality] = I

        # Compute mean I_list for each centrality type across simulations
        mean_results = {centrality: np.mean(results[centrality], axis=0) for centrality in centrality_types}
        
        return mean_results

def plot_line_graph(results, alpha=0.2, beta=1, alpha_list=None, beta_list=None, network_type=None, file_name=None): 
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
        else:
            raise ValueError 
    plt.ylabel("Mean Fraction of Failed Nodes (I)")
    plt.legend()
    plt.grid()
    plt.show()

    if file_name is not None: 
        if alpha_list is not None: 
            plt.savefig(fr'target_attack/result_graph/{file_name}_beta_{beta}.png') 
        elif beta_list is not None: 
            plt.savefig(fr'target_attack/result_graph/{file_name}_alpha_{alpha}.png') 
        else: 
            raise ValueError

def load_results_from_csv(filename):
    """
    Load simulation results from a CSV file.

    """
    
    df = pd.read_csv(filename)
    
    alpha = df["Alpha"].tolist()
    results = df.drop(columns=["Alpha"]).to_dict(orient="list")
    
    return alpha, results


def save_results_to_csv(results, filename, alpha_list=None, beta_list=None):
    """
    Save simulation results to a CSV file.

    """
    df = pd.DataFrame(results)
    if alpha_list is not None: 
        df.insert(0, "Alpha", alpha_list)
    elif beta_list is not None: 
        df.insert(0, "Beta", beta_list)
    else: 
        raise ValueError
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# load the toy network
G = nx.read_graphml("degree_distribution/toy_network_undirected.graphml")
mapping = {node: int(node) for node in G.nodes()}
G = nx.relabel_nodes(G, mapping)

# load the iceland network
G2 = nx.read_graphml("degree_distribution/iceland.graphml")
mapping2 = {node: int(node) for node in G2.nodes()}
G2 = nx.relabel_nodes(G2, mapping2)

# load the US network
G3 = nx.read_graphml("degree_distribution/us_network.graphml")
mapping3 = {node: int(node) for node in G3.nodes()}
G3 = nx.relabel_nodes(G3, mapping3)


# initialize parameters and empty lists
alpha_list = np.linspace(0,1.2,20)
beta_list = np.linspace(1,1.5,20)
alpha_value = 0.3
beta_value = 1.05
centrality_types = ["degree", "betweenness", "closeness"]

# # start simulation of the toy network with varying alpha
# results_toy_varing_alpha = simulate_and_average(G, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list)
# print(f"The result for the toy network with varing alpha is: {results_toy_varing_alpha}")
# plot_line_graph(results_toy_varing_alpha, beta=beta_value, alpha_list=alpha_list, network_type="Toy Network")
# save_results_to_csv(results_toy_varing_alpha, "toy_network_beta_results.csv", alpha_list)

# # start simulation of the iceland network with varying alpha
# results_ice_varing_alpha = simulate_and_average(G2, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list)
# print(f"The result for the iceland network with varing alpha is: {results_ice_varing_alpha}")
# plot_line_graph(results_ice_varing_alpha, beta=beta_value, alpha_list=alpha_list, network_type="Iceland Network")
# save_results_to_csv(results_ice_varing_alpha, "iceland_network_beta_results.csv", alpha_list)

# # start simulation of the US network with varying alpha
# results_us_varing_alpha = simulate_and_average(G3, centrality_types, target_attack=True, beta=beta_value, alpha_list=alpha_list)
# print(f"The result for the US network with varing alpha is: {results_us_varing_alpha}")
# plot_line_graph(results_us_varing_alpha, beta=beta_value, alpha_list=alpha_list, network_type="US Network")
# save_results_to_csv(results_us_varing_alpha, "US_network_beta_results.csv", alpha_list)

# start simulation of the toy network with varying beta
results_toy_varing_beta = simulate_and_average(G, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list)
print(f"The result for the toy network with varing beta is: {results_toy_varing_beta}")
plot_line_graph(results_toy_varing_beta, alpha=alpha_value, beta_list=beta_list, network_type="Toy Network")
save_results_to_csv(results_toy_varing_beta, "toy_network_alpha_results.csv", beta_list=beta_list)

# start simulation of the iceland network with varying beta
results_ice_varing_beta = simulate_and_average(G2, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list)
print(f"The result for the iceland network with varing beta is: {results_ice_varing_beta}")
plot_line_graph(results_ice_varing_beta, alpha=alpha_value, beta_list=beta_list, network_type="Iceland Network")
save_results_to_csv(results_ice_varing_beta, "iceland_network_alpha_results.csv", beta_list=beta_list)

# start simulation of the US network with varying beta
results_us_varing_beta = simulate_and_average(G3, centrality_types, target_attack=True, alpha=alpha_value, beta_list=beta_list)
print(f"The result for the US network with varing beta is: {results_us_varing_beta}")
plot_line_graph(results_us_varing_beta, alpha=alpha_value, beta_list=beta_list, network_type="US Network")
save_results_to_csv(results_us_varing_beta, "US_network_alpha_results.csv", beta_list=beta_list)

