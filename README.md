# No More Cascading 
Welcome to the Cascading Failure Simulation repository! This project is focused on simulating and analyzing the behavior of cascading failures in power grids, with the goal of assessing the robustness and reliability of various network configurations. By leveraging simulations of cascading failures, we explore how power grid infrastructures can respond to disruptions and how different centrality measures can influence grid resilience.

#### Project Structure

This repository contains the following key directories and files:

1. cascading_failure (Package)
utils.py: This file includes all the utility functions that are used throughout the project. It contains various helper functions for network analysis, centrality measures, and other necessary operations.
CascadingFailure.py: This is the main class of the simulation. It encapsulates the logic for performing cascading failure simulations on power grids, calculates failure propagation, and returns key results.
2. data (Directory)
This directory contains the initial data required for the simulations. The data is provided in both CSV and GraphML formats, representing the structure and characteristics of various power grids.

CSV files: Contain network node and edge data in tabular format.
GraphML files: Represent the graph structure of power grids in XML format, including node and edge information.
3. results (Directory)
The results of the simulations are stored here. After running the cascading failure simulations, results are generated and saved in both graphical and tabular formats.

Plots (PNG): Visualization of the simulation results, typically showing the relationship between various parameters like centrality measures, failure fractions, and network capacity.
CSV files: Store simulation data that can later be used to regenerate plots without needing to rerun the simulations. This allows for more efficient analysis and comparison.
