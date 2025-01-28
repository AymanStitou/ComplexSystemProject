import matplotlib.pyplot as plt
import logging
import utilities as utils
def plot_degree_dists(G_degrees, ER_degrees):
    logging.info('Plotting the degree distribution histograms')
    plt.figure(figsize=(10, 6))
    

    plt.hist(G_degrees, bins=range(min(G_degrees), max(G_degrees) + 2), density=True, alpha=0.5, label="Original Graph", color='blue', edgecolor='black')
    plt.hist(ER_degrees, bins=range(min(ER_degrees), max(ER_degrees) + 2), density=True, alpha=0.5, label="Erdos–Renyi", color ='green', edgecolor='black')

    # Adding labels and title
    plt.xlabel("Degree")
    plt.ylabel("Fraction of Nodes")
    plt.title("Degree Distribution Comparison (Histogram)")

    # Adding legend and grid
    plt.legend()
    plt.grid(False)

    # Showing the plot
    plt.show()

def plot_fraction_failed_nodes(alpha, results, network_name = '', save_results = False, save_plot = False):
    plt.figure(figsize=(10, 6))
    for centrality, mean_result in results.items():
        plt.plot(alpha, mean_result, label=f"{centrality.capitalize()} Centrality", marker='o')
    plt.title(f"Mean Fraction of Failed Nodes vs. Alpha {network_name} Network")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Fraction of Failed Nodes (I)")
    plt.legend()
    plt.grid()
    plt.show()
    
    if save_plot:
        plot_filename = f'results/plots/{network_name}_fraction_failed_nodes.png'
        plt.savefig(plot_filename)  
        logging.info(f"Plot saved")

    
    if save_results:
        utils.save_results_to_csv(results, alpha, f'results/{network_name}results.csv' )
