Role: You are an expert Performance Engineer and Senior Deep Learning Researcher. You excel in Python, CUDA, and modern JAX-based frameworks (specifically DeepMind's Haiku and Jraph). You have deep domain expertise in Graph Neural Networks (GNNs), Diffusion Models, and AI for weather modeling (e.g., handling multi-dimensional xarray datasets).

Task: Write a robust, production-ready benchmark script (main.py) to measure and compare the inference performance and accuracy of Google DeepMind's GenCast and GraphCast models.

Configuration Management:

The script must strictly use Hydra for configuration management.

All experiment variables must be loaded from a config/ directory (e.g., config/config.yaml).

Test Data: - ERA5 (Historical reanalysis) and HRES (High-Resolution Forecast).

The Hydra config will specify the model, dataset version, specific atmospheric/surface variables, and time slices to load.

Test Models: - Support all available pre-trained models within the deepmind/graphcast repository (referencing their CONTRIBUTING.md and README.md for proper checkpoint loading).

Execution Flow:

Initialize: Read the configuration via Hydra.

Instantiate: Build the model architecture (GraphCast or GenCast) dynamically based on the configured model type.

Load Weights: Load the pre-trained parameters into the Haiku model state.

Load & Preprocess Data: Load the specified ERA5/HRES data using xarray. Apply the necessary normalization and preprocessing required by the respective model architectures.

Inference (The Benchmark): - Crucial Distinction: Handle the inference loop differently based on the model. GraphCast is a deterministic GNN (single trajectory). GenCast is a probabilistic diffusion model (requires solving the diffusion SDE to generate an ensemble of forecasts).

JAX Timing: You must use jax.block_until_ready() to accurately record inference time, accounting for JAX's asynchronous dispatch. Include a warm-up step to exclude JAX compilation (jax.jit) time from the benchmark.

Compute Metrics: - Compute CRPS (Continuous Ranked Probability Score, NRMSE) for GenCast's ensemble outputs.

Compute standard deterministic metrics (NRMSE, MAE) for GraphCast.

Record total inference time, memory footprint (if possible), and throughput.

Output: Log the metrics to the console and conditionally save qualitative results (e.g., plotting the forecast grids) if a save_visualizations flag is set to true in the Hydra config.

Constraints:

Reference the benchmark flow found in gencast_demo_cloud_vm.ipynb; gencast_mini_demo.ipynb, and graphcaset_demo.ipynb

Ensure type hints are used throughout the code.

Write modular code: separate data loading, model instantiation, and evaluation into clean, helper functions.

The TPU setting is not available please set up all the code with Nvidia GPU CUDA.

