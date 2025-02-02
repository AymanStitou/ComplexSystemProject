import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from .CascadingFailure import CascadingFailureSimulation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional

def load_network(filepath: str) -> nx.Graph:
    """
    Load a network from a GraphML file and relabel nodes as integers.

    Args:
        filepath (str): Path to the GraphML file.

    Returns:
        nx.Graph: A NetworkX graph with integer-labeled nodes.
    """
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
    return G

def load_attack_results_from_csv(filename: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Load simulation results from a CSV file.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        Tuple[List[float], Dict[str, List[float]]]: A tuple containing a list of alpha values and a dictionary of results.
    """
    df = pd.read_csv(filename)
    alpha = df["Alpha"].tolist()
    results = df.drop(columns=["Alpha"]).to_dict(orient="list")
    return alpha, results

def initialize_simulation(G: nx.Graph) -> CascadingFailureSimulation:
    """
    Initialize a cascading failure simulation for a given graph.

    Args:
        G (nx.Graph): The network graph.

    Returns:
        CascadingFailureSimulation: An initialized simulation instance.
    """
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()
    return simulation

def simulation_capacity(
    initial_failures: List[int],
    centrality: str,
    simulation: CascadingFailureSimulation,
    capacity_list: List[float]
) -> List[float]:
    """
    Run a cascading failure simulation for different capacity values.

    Args:
        initial_failures (List[int]): List of initially failed nodes.
        centrality (str): Centrality measure used for capacity allocation.
        simulation (CascadingFailureSimulation): The simulation instance.
        capacity_list (List[float]): List of total capacity values.

    Returns:
        List[float]: List of fraction of failed nodes for each capacity value.
    """
    sum_centrality = simulation.calculate_centrality_measures()
    I_list = []
    for c in capacity_list:
        simulation.calculate_initial_load(centrality_type=centrality, sum_centrality=sum_centrality)
        simulation.calculate_capacity(total_capacity=c)
        _, _, I, _ = simulation.simulate_cascading_failure(initial_failures)
        I_list.append(I)
    return I_list

def run_prevention_mechanism_simulations(
    simulation: CascadingFailureSimulation,
    alpha_values: List[float],
    prevention_mechanisms: List[str],
    num_simulations: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run cascading failure simulations with different prevention mechanisms.

    Args:
        simulation (CascadingFailureSimulation): The simulation instance.
        alpha_values (List[float]): List of alpha values.
        prevention_mechanisms (List[str]): List of prevention mechanisms.
        num_simulations (int): Number of simulations to run.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Dictionary containing simulation results for each prevention mechanism.
    """
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

def run_target_attack_simulations(
    initial_failures: List[int],
    centrality_type: str,
    simulation: CascadingFailureSimulation,
    alpha: float = 0.2,
    beta: float = 1.0,
    alpha_list: Optional[List[float]] = None,
    beta_list: Optional[List[float]] = None,
    use_prevention: bool = False
) -> List[float]:
    """
    Run the cascading failure simulation for a specific network and centrality measure.
    It returns a list of the number of failed nodes for each alpha or beta value.

    Args:
        initial_failures (List[int]): List of initially failed nodes.
        centrality_type (str): Centrality measure used.
        simulation (CascadingFailureSimulation): The simulation instance.
        alpha (float, optional): Default alpha value. Defaults to 0.2.
        beta (float, optional): Default beta value. Defaults to 1.0.
        alpha_list (Optional[List[float]], optional): List of alpha values. Defaults to None.
        beta_list (Optional[List[float]], optional): List of beta values. Defaults to None.
        use_prevention (bool, optional): Whether to use a prevention mechanism. Defaults to False.

    Returns:
        List[float]: List of the number of failed nodes for each alpha or beta value.
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

def run_simulation_single_pair(
    G: nx.Graph,
    alpha: float,
    beta: float,
    initial_failures: List[int],
    centrality_type: str,
    simulation: CascadingFailureSimulation
) -> float:
    """
    Run a single cascading failure simulation for given parameters and return the fraction of failed nodes.

    Args:
        G (nx.Graph): The network graph.
        alpha (float): Alpha value for capacity calculation.
        beta (float): Beta value for capacity calculation.
        initial_failures (List[int]): List of initially failed nodes.
        centrality_type (str): Centrality measure used.
        simulation (CascadingFailureSimulation): The simulation instance.

    Returns:
        float: Fraction of failed nodes in the network.
    """
    simulation.calculate_initial_load(centrality_type=centrality_type)
    simulation.calculate_capacity(alpha=alpha, beta=beta)
    failed_nodes, _, _, _ = simulation.simulate_cascading_failure(initial_failures)
    return len(failed_nodes) / len(G)

def run_simulation(
    G: nx.Graph,
    alpha: List[float],
    initial_failures: List[int],
    centrality_type: str,
    simulation: CascadingFailureSimulation,
    beta: float = 1.0
) -> List[float]:
    """
    Run cascading failure simulations for different alpha values.

    Args:
        G (nx.Graph): The network graph.
        alpha (List[float]): List of alpha values.
        initial_failures (List[int]): List of initially failed nodes.
        centrality_type (str): Centrality measure used.
        simulation (CascadingFailureSimulation): The simulation instance.
        beta (float, optional): Beta value for capacity calculation. Defaults to 1.0.

    Returns:
        List[float]: List of the fraction of failed nodes for each alpha value.
    """
    n_failed_nodes = []
    I_list = []

    for a in alpha:
        simulation.calculate_initial_load(centrality_type=centrality_type)
        simulation.calculate_capacity(alpha=a, beta=beta)
        failed_nodes = simulation.simulate_cascading_failure(initial_failures)
        n_failed_nodes.append(len(failed_nodes))
        I_list.append(len(failed_nodes) / len(G))

    return I_list

def simulate_and_average_capacity(
    G: nx.Graph,
    centrality_types: List[str],
    capacity_list: List[float],
    num_simulations: int = 25,
    target_attack: bool = False
) -> Dict[str, List[float]]:
    """
    Simulate cascading failures multiple times and compute average failure impact across different centrality types.

    Args:
        G (nx.Graph): The network graph.
        centrality_types (List[str]): List of centrality measures to evaluate.
        capacity_list (List[float]): List of capacity values to simulate.
        num_simulations (int, optional): Number of simulations to average over. Defaults to 25.
        target_attack (bool, optional): If True, use targeted attacks based on ranked centrality. Defaults to False.

    Returns:
        Dict[str, List[float]]: Dictionary mapping each centrality type to the average failure impact.
    """
    results = {centrality: [] for centrality in centrality_types}
    total_nodes = len(G.nodes)
    num_failures = max(1, int(total_nodes * 0.01))
    simulation = CascadingFailureSimulation(G)
    
    if target_attack:
        for centrality in centrality_types:
            initial_failures = simulation.rank_centrality(centrality, num_failures)
            I = simulation_capacity(initial_failures, centrality, simulation, capacity_list)
            results[centrality] = I
            print(f"Finish simulation of the centrality type: {centrality}")
        return results
    else:
        for _ in range(num_simulations):
            initial_failures = random.sample(range(1, total_nodes - 1), num_failures)
            for centrality in centrality_types:
                I = simulation_capacity(initial_failures, centrality, simulation, capacity_list)
                results[centrality].append(I)
                print(f"Finish simulation of the centrality type: {centrality}")

        mean_results = {centrality: np.mean(results[centrality], axis=0) for centrality in centrality_types}
        return mean_results


def load_network(filepath: str) -> nx.Graph:
    """
    Load a network from a GraphML file and relabel nodes as integers.

    Args:
        filepath (str): Path to the GraphML file.

    Returns:
        nx.Graph: A NetworkX graph with integer-labeled nodes.
    """
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
    return G

def load_attack_results_from_csv(filename: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Load simulation results from a CSV file.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        Tuple[List[float], Dict[str, List[float]]]: A tuple containing a list of alpha values and a dictionary of results.
    """
    df = pd.read_csv(filename)
    alpha = df["Alpha"].tolist()
    results = df.drop(columns=["Alpha"]).to_dict(orient="list")
    return alpha, results

def initialize_simulation(G: nx.Graph) -> CascadingFailureSimulation:
    """
    Initialize a cascading failure simulation for a given graph.

    Args:
        G (nx.Graph): The network graph.

    Returns:
        CascadingFailureSimulation: An initialized simulation instance.
    """
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()
    return simulation

def simulate_and_average(
    G: nx.Graph,
    alpha: List[float],
    centrality_types: List[str],
    target_attack: bool = False,
    num_simulations: int = 30,
    beta: float = 1.2
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate cascading failures multiple times and compute mean and standard deviation of failure impact.

    Args:
        G (nx.Graph): The network graph.
        alpha (List[float]): List of alpha values.
        centrality_types (List[str]): List of centrality measures to evaluate.
        target_attack (bool, optional): If True, use targeted attacks based on ranked centrality. Defaults to False.
        num_simulations (int, optional): Number of simulations to average over. Defaults to 30.
        beta (float, optional): Beta value for capacity calculation. Defaults to 1.2.

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary mapping each centrality type to (mean impact, std deviation).
    """
    results = {centrality: [] for centrality in centrality_types}
    total_nodes = len(G.nodes)
    simulation = CascadingFailureSimulation(G)
    simulation.calculate_centrality_measures()
    num_failures = max(1, int(total_nodes * 0.01))
    
    if target_attack:
        for centrality in centrality_types:
            initial_failures = simulation.rank_centrality(centrality, num_failures)
            I = run_simulation(G, alpha, initial_failures, centrality, simulation, beta=beta)
            results[centrality] = (I, np.zeros_like(I))
            return results
    else:
        for _ in range(num_simulations):
            initial_failures = random.sample(range(1, total_nodes - 1), num_failures)
            for centrality in centrality_types:
                I = run_simulation(G, alpha, initial_failures, centrality, simulation, beta=beta)
                results[centrality].append(I)

        mean_results = {centrality: np.mean(results[centrality], axis=0) for centrality in centrality_types}
        std_results = {centrality: np.std(results[centrality], axis=0, ddof=1) for centrality in centrality_types}
        return {centrality: (mean_results[centrality], std_results[centrality]) for centrality in centrality_types}


def simulate_and_average_3D(
    G: any, 
    alpha_values: List[float], 
    beta_values: List[float], 
    centrality_types: List[str], 
    num_simulations: int = 5, 
    p_fail: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Simulates cascading failures on a network graph for various alpha and beta values, 
    and averages the results over multiple simulations. Results are returned as a 3D structure 
    for each centrality type, indexed by beta and alpha values.

    Args:
        G (any): The input graph on which the simulation will be run. This can be a networkx graph or 
                 another compatible graph object.
        alpha_values (List[float]): A list of alpha values to iterate over during the simulation.
        beta_values (List[float]): A list of beta values to iterate over during the simulation.
        centrality_types (List[str]): A list of centrality types (e.g., "degree", "betweenness", etc.) 
                                      to use in the simulations.
        num_simulations (int, optional): The number of simulations to run for each (alpha, beta) pair.
                                          Defaults to 5.
        p_fail (float, optional): The fraction of nodes in the graph that initially fail. Defaults to 0.01 
                                  (1% of the nodes).

    Returns:
        Dict[str, np.ndarray]: A dictionary where the keys are centrality types and the values are 2D arrays.
                                Each array corresponds to the average fractional failure for each combination 
                                of alpha and beta values, with shape (len(beta_values), len(alpha_values)).
    """
    simulation = CascadingFailureSimulation(G)
    sum_centrality = simulation.calculate_centrality_measures()  # Get sum of centralities

    # Initialize the results dictionary for storing fractional failures
    results_3D = {
        cent: np.zeros((len(beta_values), len(alpha_values))) 
        for cent in centrality_types
    }

    total_nodes = len(G)
    n_failures = max(1, int(total_nodes * p_fail))  # At least one node should fail

    # Iterate over the beta values
    for j, b in enumerate(beta_values):
        
        print(f"Currently at {(j/len(beta_values)) * 100}%")
        # Iterate over the alpha values
        for i, a in enumerate(alpha_values):
            
            # Iterate over the centrality types
            for cent in centrality_types:
                frac_acc = 0.0
                
                # Run the simulation multiple times for each combination
                for _ in range(num_simulations):
                    initial_failures = random.sample(list(G.nodes()), n_failures)
                    frac_acc += run_simulation_single_pair(G, a, b, initial_failures, cent, simulation)
                
                # Store the average fractional failure for this centrality, alpha, and beta
                results_3D[cent][j, i] = frac_acc / num_simulations

    return results_3D

def plot_prevention_mechanism_results(
    results: Dict[str, Dict[str, List[float]]], 
    alpha_values: List[float], 
    prevention_mechanisms: List[str], 
    num_simulations: int, 
    saveplot: bool
) -> None:
    """
    Plots the results of a cascading failure simulation for different prevention mechanisms.
    
    The function generates two plots:
    1. Cascading Failure (CF) vs Average Total Capacity
    2. Fraction of Failed Nodes (I) vs Average Total Capacity
    
    Each plot is averaged over the given number of simulations and can be optionally saved as an image.
    
    Args:
        results (Dict[str, Dict[str, List[float]]]): A dictionary containing simulation results, where the keys are 
                                                    prevention mechanisms and the values are dictionaries with 'total_capacity', 
                                                    'CF', and 'I' as keys for the respective results.
        alpha_values (List[float]): A list of alpha values used in the simulation. This is not directly used in the plotting, 
                                     but is passed for context.
        prevention_mechanisms (List[str]): A list of strings representing the names of the prevention mechanisms to plot.
        num_simulations (int): The number of simulations used for averaging.
        saveplot (bool): If True, the plots will be saved to disk as images.

    Returns:
        None: The function does not return any value; it generates and optionally saves plots.
    """
    markers = ["o", "s", "D", "^", "x"]
    line_styles = ["-", "--", "-.", ":", "-"]
    
    # Plot Cascading Failure (CF) vs Average Total Capacity
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

    # Plot Fraction of Failed Nodes (I) vs Average Total Capacity
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
        plt.savefig('results/plots/prevention_mechanism_I.png')
    plt.show()

def save_prevention_results_to_csv(
    results: Dict[str, Dict[str, List[float]]], 
    alpha_values: List[float], 
    num_simulations: int, 
    filename: str
) -> None:
    """
    Saves the results of a cascading failure simulation for different prevention mechanisms to a CSV file.
    
    The function creates a DataFrame where each prevention mechanism has columns for "CF" (Cascading Failure) and "I" 
    (Fraction of Failed Nodes) values, along with the corresponding "alpha" values.

    Args:
        results (Dict[str, Dict[str, List[float]]]): A dictionary containing simulation results, where the keys are 
                                                    prevention mechanisms and the values are dictionaries with 'CF' 
                                                    and 'I' as keys for the respective results.
        alpha_values (List[float]): A list of alpha values used in the simulation.
        num_simulations (int): The number of simulations used for averaging.
        filename (str): The name of the file to save the results as a CSV.

    Returns:
        None: The function does not return any value; it saves the results to a CSV file.
    """
    # Create a DataFrame with alpha values as the first column
    df = pd.DataFrame({"alpha": alpha_values})

    # Add columns for each prevention mechanism's "CF" and "I" values
    for mechanism in results:
        df[f"CF_{mechanism}"] = results[mechanism]["CF"]
        df[f"I_{mechanism}"] = results[mechanism]["I"]

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot_attack_graphs(
    results: Dict[str, List[float]], 
    alpha: float = 0.2, 
    beta: float = 1, 
    alpha_list: Optional[List[float]] = None, 
    beta_list: Optional[List[float]] = None, 
    capacity_list: Optional[List[float]] = None, 
    network_type: Optional[str] = None, 
    file_name: Optional[str] = None
) -> None:
    """
    Plots the mean fraction of failed nodes for different centrality measures (e.g., degree, betweenness) 
    as a function of varying parameters (alpha, beta, or capacity) across multiple network types.

    The function can plot the results for:
    1. Alpha values, with beta fixed.
    2. Beta values, with alpha fixed.
    3. Capacity values, with no specific alpha or beta.

    Args:
        results (Dict[str, List[float]]): A dictionary where the keys are centrality measures (e.g., "degree", "betweenness")
                                          and the values are lists of mean fraction of failed nodes corresponding to 
                                          varying parameters (alpha, beta, or capacity).
        alpha (float, optional): The alpha value for fixed parameters, default is 0.2.
        beta (float, optional): The beta value for fixed parameters, default is 1.
        alpha_list (Optional[List[float]], optional): A list of alpha values for varying the alpha parameter. Default is None.
        beta_list (Optional[List[float]], optional): A list of beta values for varying the beta parameter. Default is None.
        capacity_list (Optional[List[float]], optional): A list of capacity values for varying the network's capacity. Default is None.
        network_type (Optional[str], optional): A string representing the network type (e.g., "scale-free", "random"). Default is None.
        file_name (Optional[str], optional): The filename to save the plot. If None, the plot will be shown instead of saved. Default is None.

    Returns:
        None: The function generates a plot and optionally saves it to a file.
    """
    plt.figure(figsize=(10, 6))
    
    # Loop through each centrality measure and its corresponding result
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
            raise ValueError("No input of varying variables (alpha/beta/capacity)")

    plt.ylabel("Mean Fraction of Failed Nodes (I)")
    plt.legend()
    plt.grid()

    # Save the plot if file_name is provided, otherwise show the plot
    if file_name is not None: 
        if alpha_list is not None: 
            plt.savefig(fr'results/plots/{file_name}_beta_{beta}.png') 
        elif beta_list is not None: 
            plt.savefig(fr'results/plots/{file_name}_alpha_{alpha}.png') 
        elif capacity_list is not None:
            plt.savefig(fr'results/plots/{file_name}.png') 
        else: 
            raise ValueError("No input of varying variables (alpha/beta/capacity)")
    else: 
        plt.show()
        
def plot_3D_results_from_csv(filename: str) -> None:
    """
    Plots 3D surface plots of the fraction of failed nodes (I) for different centrality measures 
    as a function of alpha and beta values, based on data from a CSV file.

    The CSV file is expected to have the following columns:
    - 'centrality': The centrality measure used (e.g., "degree", "betweenness").
    - 'alpha': The alpha value used in the simulation.
    - 'beta': The beta value used in the simulation.
    - 'I': The fraction of failed nodes for the given centrality, alpha, and beta.

    Args:
        filename (str): The path to the CSV file containing the simulation results.

    Returns:
        None: The function generates 3D plots and displays them.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Get unique centrality types and sort the alpha and beta values
    centrality_types = df['centrality'].unique()
    alpha_vals = np.sort(df['alpha'].unique())
    beta_vals = np.sort(df['beta'].unique())
    
    # Create meshgrid for alpha and beta values
    A, B = np.meshgrid(alpha_vals, beta_vals)
    
    # Number of centralities to create subplots for
    num_centralities = len(centrality_types)
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(6 * num_centralities, 6))  

    # Loop through each centrality type and plot a 3D surface
    for idx, cent in enumerate(centrality_types, start=1):
        ax = fig.add_subplot(1, num_centralities, idx, projection='3d')
        
        # Pivot the DataFrame to create a matrix for 'I' values
        pivot_df = df[df['centrality'] == cent].pivot(index='beta', columns='alpha', values='I')

        Z = pivot_df.values  # Z values for the surface plot
        surf = ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Set plot titles and labels
        ax.set_title(f"Centrality: {cent.capitalize()}", fontsize=14)
        ax.set_xlabel(r"$\alpha$", fontsize=12)
        ax.set_ylabel(r"$\beta$", fontsize=12)
        ax.set_zlabel("Fraction Failed (I)", fontsize=12)
        
        # Adjust the viewing angle
        ax.view_init(elev=25, azim=60)  

    # Adjust layout to prevent overlap and display the plot
    plt.tight_layout()
    plt.show()
def save_attack_results_to_csv(
    results: dict, 
    filename: str, 
    alpha_list: Optional[List[float]] = None, 
    beta_list: Optional[List[float]] = None, 
    capacity_list: Optional[List[float]] = None
) -> None:
    """
    Saves the simulation results to a CSV file, adding columns for alpha, beta, or total capacity 
    based on the provided lists.

    Args:
        results (dict): A dictionary containing simulation results. The keys should be column names,
                         and the values should be lists of results.
        filename (str): The name of the file to save the results to.
        alpha_list (Optional[List[float]], optional): A list of alpha values for the simulation. Default is None.
        beta_list (Optional[List[float]], optional): A list of beta values for the simulation. Default is None.
        capacity_list (Optional[List[float]], optional): A list of total capacity values for the simulation. Default is None.

    Returns:
        None: The function saves the results to a CSV file and prints a confirmation message.
    """
    # Create DataFrame from the results dictionary
    df = pd.DataFrame(results)

    # Add the appropriate column (alpha, beta, or capacity)
    if alpha_list is not None: 
        df.insert(0, "Alpha", alpha_list)
    elif beta_list is not None: 
        df.insert(0, "Beta", beta_list)
    elif capacity_list is not None: 
        df.insert(0, "Total_Capacity", capacity_list)
    else: 
        raise ValueError("No input of varying variables (alpha/beta/capacity)")
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def save_results_3D_to_csv(
    results_3D: Dict[str, np.ndarray], 
    alpha_vals: List[float], 
    beta_vals: List[float], 
    filename: str
) -> None:
    """
    Saves the 3D simulation results (centrality, alpha, beta, and fraction of failed nodes) to a CSV file.
    
    Args:
        results_3D (Dict[str, np.ndarray]): A dictionary where the keys are centrality types (e.g., "degree", "betweenness") 
                                             and the values are 2D numpy arrays representing the simulation results.
        alpha_vals (List[float]): A list of alpha values.
        beta_vals (List[float]): A list of beta values.
        filename (str): The path to save the results CSV file.

    Returns:
        None: The function saves the results to a CSV file and prints a confirmation message.
    """
    rows = []

    # Loop through each centrality and its corresponding result matrix
    for cent, matrix in results_3D.items():
        for j, b in enumerate(beta_vals):
            for i, a in enumerate(alpha_vals):
                rows.append({
                    "alpha": a,
                    "beta": b,
                    "I": matrix[j, i],
                    "centrality": cent
                })

    # Create DataFrame from rows
    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")


