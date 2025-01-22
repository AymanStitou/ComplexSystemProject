from CascadingFailure import CascadingFailureSimulation


simulation = CascadingFailureSimulation(num_nodes=50, edges_per_node=3)


simulation.calculate_centrality_measures()


simulation.calculate_initial_load()


simulation.calculate_capacity(alpha=0.2, beta=1)


initial_failures = [0]  
failed_nodes = simulation.simulate_cascading_failure(initial_failures)


simulation.visualize_network(failed_nodes)


print("Failed nodes:", failed_nodes)

simulation.print_centrality_measures()