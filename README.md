# No More Cascading: Cascading Failure Simulation

Welcome to the **No More Cascading** repository! This project focuses on simulating and analyzing cascading failures in power grids. By leveraging network analysis and failure propagation models, we assess the robustness and reliability of various grid configurations. Our goal is to understand how power grid infrastructures respond to disruptions and how different centrality measures influence resilience.

## Project Structure

This repository is organized into several key directories and files:

### 1. `cascading_failure/`

This directory contains the core simulation logic and utility functions:

- **`utils.py`**: Utility functions for network analysis, centrality measures, and general operations used throughout the project.
- **`CascadingFailure.py`**: The main class responsible for performing cascading failure simulations. It calculates failure propagation and returns key results.

### 2. `data/`

This directory contains the initial datasets required for simulations. The data is provided in multiple formats:

- **CSV files**: Tabular representations of network nodes and edges.
- **GraphML files**: XML-based representations of power grid graphs, including node and edge attributes.

### 3. `results/`

This directory stores the outputs of the simulations, allowing for further analysis without re-running the experiments:

- **Plots (****`.png`****)**: Visualizations of simulation results, such as relationships between centrality measures, failure fractions, and network capacity.
- **CSV files**: Tabular simulation results for efficient analysis and comparison.

### 4. `analysis.ipynb`

This Jupyter Notebook is the primary workspace for conducting experiments, running simulations, and analyzing results. All experiments are documented and executed within this file.

### 5. `old/`

This directory contains deprecated code used throughout the project but is no longer active in `analysis.ipynb`. It serves as an archive of previous implementations.

---

## Getting Started

To get started with the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/no-more-cascading.git
   ```
2. Install the necessary dependencies (if applicable).
3. Open `analysis.ipynb` and run the notebook to simulate cascading failures.

For further details on implementation and methodology, explore the `cascading_failure/` directory and refer to the Jupyter Notebook.
