from CascadingFailure import CascadingFailureSimulation
import random as ra
import matplotlib.pyplot as plt

simulation = CascadingFailureSimulation()


simulation.calculate_centrality_measures()


alpha = [0,0.005,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
beta = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
n_fail_nodes = []
CF_list = []
for a in alpha:
    simulation.calculate_centrality_measures()
    simulation.calculate_initial_load(centrality_type='degree')
    simulation.calculate_capacity(alpha=a, beta=1.2)

    #fail = ra.randint(0, 24)
    #fail1 = ra.randint(0, 24)
    #fail2 = ra.randint(0, 24)
    #initial_failures = [fail, fail1, fail2] 
    initial_failures = [11] 
    failed_nodes, CF = simulation.simulate_cascading_failure(initial_failures)
    n_fail_nodes.append(len(failed_nodes))
    CF_list.append(CF)
print(CF_list)
plt.scatter(alpha, CF_list, color = "red")
plt.plot(alpha, CF_list, color = "blue")
plt.xlabel("value of alpha")
plt.ylabel("CF")
plt.grid()
plt.show()


simulation.visualize_network(failed_nodes)


print("Failed nodes:", failed_nodes)

simulation.print_centrality_measures()


