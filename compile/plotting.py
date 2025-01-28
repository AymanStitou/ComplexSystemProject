import matplotlib.pyplot as plt
import logging
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
