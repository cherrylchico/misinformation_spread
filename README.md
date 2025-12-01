# Misinformation Diffusion Simulation

This project simulates the spread of misinformation in social networks using agent-based modeling. It supports flexible configuration for agents, influencers, and bots, and provides visualizations for network structure and diffusion dynamics.

## 1. Environment Setup

First, create and activate the Python environment using [uv](https://github.com/astral-sh/uv):

```sh
uv sync
source .venv/bin/activate
```

## 2. Create the Social Network

Run the network creation script to generate a network for simulation. Use the following command with required and recommended options:

```sh
uv run python code/create_network.py 100 facebook --sampling_method community --network_file data/facebook_combined.txt --save
```

- `100` : Number of agents (required)
- `facebook` : Network type (required)
- `--sampling-method community` : Use community-based sampling
- `--save` : Save the generated network to a pickle file

This will create a network pickle file in the `data/` directory.

## 3. Run the Simulation

Use the main simulation script to run experiments. You can specify different agent configurations:

### Baseline (Regular Agents Only)
```sh
uv run python code/run_simulation.py \
  --config-sim code/config/config_sim_general.json \
  --config-agent code/config/config_sim_agent.json \
  --no-gif
```

### With Influencers
```sh
uv run python code/run_simulation.py \
  --config-sim code/config/config_sim_general.json \
  --config-agent code/config/config_sim_agent.json \
  --config-influencer code/config/config_sim_influencer.json
```

### With Influencers and Bots
```sh
uv run python code/run_simulation.py \
  --config-sim code/config/config_sim_general.json \
  --config-agent code/config/config_sim_agent.json \
  --config-influencer code/config/config_sim_influencer.json \
  --config-bot code/config/config_sim_bot.json \
  --no-gif \
  --seed 42
```

### Additional Options

- `--no-gif`: Skip GIF generation to save time (useful for quick test runs)

```sh
uv run python code/run_simulation.py \
  --config-sim code/config/config_sim_general.json \
  --config-agent code/config/config_sim_agent.json \
  --no-gif
```

## 4. Output

Simulation results and visualizations are automatically saved in timestamped directories under `output/`. Each run creates a new folder with the format `output/YYYYMMDD_HHMMSS_output/` containing:
- Network visualizations (political bias, initial reputation, message diffusion)
- Metrics plots (reach, forwarding rate, misinformation contamination, reputation)
- Animated GIF of message diffusion (optional, skip with `--no-gif`)

Example output directory: `output/20251129_164513_output/`

## 5. Configuration

Edit the JSON files in `code/config/` to customize agent, influencer, bot, and simulation parameters.


# Reproducibility

By default, each simulation run uses random sampling, resulting in different outcomes. To get reproducible results:

```sh
uv run python code/run_simulation.py \
  --config-sim code/config/config_sim_general.json \
  --config-agent code/config/config_sim_agent.json \
  --seed 42
```

The `--seed` option sets the random seed, ensuring identical results across runs with the same parameters.

## 6. Batch Simulations & Analysis

To run multiple simulations, vary parameters, and analyze results statistically, use the batch simulation system.

### Running Batch Simulations
Use `code/batch_simulation.py` to run experiments.

**Example: Varying Probability of Truth**
```sh
uv run python code/batch_simulation.py \
  --num-runs 50 \
  --vary message.prob_truth \
  --values 0.1 0.3 0.5 0.7 0.9 \
  --config-sim code/config/config_sim_general.json \
  --config-agent code/config/config_sim_agent.json \
  --output results/prob_truth_experiment.csv
```

**Key Arguments:**
- `--num-runs`: Number of simulations per parameter value
- `--vary`: Parameter to vary (e.g., `message.prob_truth`, `num_bots`, `num_influencers`)
- `--values`: List of values to test
- `--output`: Output CSV file path

### Analyzing Results
Use `code/analyze_results.py` to generate plots and statistics.

```sh
uv run python code/analyze_results.py \
  --input results/prob_truth_experiment.csv \
  --output figures/prob_truth_analysis.png
```

This will generate a plot with error bars (95% confidence intervals) showing how metrics change with the varied parameter.

### New Parameters
- `truth_revelation_prob`: Probability that the truth of a message is revealed after each round (default: 1.0). Can be set in `config_sim_general.json` under `message`.
