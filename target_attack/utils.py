from CascadingFailure import CascadingFailureSimulation
import matplotlib.pyplot as plt 
import random
import pandas as pd
import numpy as np

def run_simulation(initial_failures, centrality_type, simulation, alpha=0.2, beta=1, alpha_list=None, beta_list=None, use_prevention=False):
    """
    Run the cascading failure simulation for a specific network and centrality measure.
    It returns a list of the number of failed nodes for each alpha value.
    """
    I_list = []

    if alpha_list is not None: 
        for a in alpha_list:
            simulation.calculate_initial_load(centrality_type=centrality_type)
            simulation.calculate_capacity(alpha=a, beta=beta)  
            _, _, I, _ = simulation.simulate_cascading_failure(initial_failures, use_prevention=use_prevention)
            I_list.append(I)
    
    if beta_list is not None: 
        for b in beta_list: 
            simulation.calculate_initial_load(centrality_type=centrality_type)
            simulation.calculate_capacity(alpha=alpha, beta=b) 
            _, _, I, _ = simulation.simulate_cascading_failure(initial_failures, use_prevention=use_prevention)
            I_list.append(I)

    return I_list

def simulate_and_average(G, centrality_types, num_simulations=25, target_attack=False, alpha=0.2, beta=1, alpha_list=None, beta_list=None, use_prevention=False):
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
                I = run_simulation(initial_failures, centrality, simulation, alpha_list=alpha_list, beta=beta, use_prevention=use_prevention)
            elif beta_list is not None: 
                I = run_simulation(initial_failures, centrality, simulation, beta_list=beta_list, alpha=alpha, use_prevention=use_prevention)
            else: 
                raise ValueError("No input of varying variables (alpha/beta)")
            results[centrality] = I
            print(fr"Finish simulation of the centrality type: {centrality}")

        return results
    
    else: 
        for _ in range(num_simulations, attacked_type='random'):
            initial_failures = random.sample(range(1,total_nodes-1), num_failures)
            for centrality in centrality_types:
                if alpha_list is not None: 
                    I = run_simulation(initial_failures, centrality, simulation, alpha_list=alpha_list, beta=beta, use_prevention=use_prevention)
                elif beta_list is not None: 
                    I = run_simulation(initial_failures, centrality, simulation, beta_list=beta_list, alpha=alpha, use_prevention=use_prevention)
                else: 
                    raise ValueError("No input of varying variables (alpha/beta)")
                results[centrality] = I
                print(fr"Finish simulation of the centrality type: {centrality}")

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
            raise ValueError("No input of varying variables (alpha/beta)")
    plt.ylabel("Mean Fraction of Failed Nodes (I)")
    plt.legend()
    plt.grid()

    if file_name is not None: 
        if alpha_list is not None: 
            plt.savefig(fr'target_attack/result_graph/{file_name}_beta_{beta}.png') 
        elif beta_list is not None: 
            plt.savefig(fr'target_attack/result_graph/{file_name}_alpha_{alpha}.png') 
        else: 
            raise ValueError("No input of varying variables (alpha/beta)")
    else: 
        plt.show()

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
        raise ValueError("No input of varying variables (alpha/beta)")
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")