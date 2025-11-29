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
  --config-agent code/config/config_sim_agent.json
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
  --config-bot code/config/config_sim_bot.json
```

## 4. Output

Simulation results and visualizations are automatically saved in timestamped directories under `output/`. Each run creates a new folder with the format `output/YYYYMMDD_HHMMSS_output/` containing:
- Network visualizations (political bias, initial reputation, message diffusion)
- Metrics plots (reach, forwarding rate, misinformation contamination, reputation)
- Animated GIF of message diffusion

Example output directory: `output/20251129_164513_output/`

## 5. Configuration

Edit the JSON files in `code/config/` to customize agent, influencer, bot, and simulation parameters.

