# Misinformation Diffusion Simulation

This code implements the misinformation diffusion model in social networks described in the paper, with interactive visualizations.

## Main Features

- **Two types of agents**:
  - **Influencers (high_reputation)**: Few agents (2-4) that care a lot about reputation. They are the ones who start the news.
  - **Regular users (low_reputation)**: Majority of agents that don't care much about reputation (parameter 0.5).

- **Visualizations**:
  - Static graphs of the network and metrics
  - **Animated GIF** showing diffusion round by round

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### Using Facebook Network (Default)

Run the simulation with default parameters:

```bash
python3 simulation.py
```

This uses the Facebook network by default. You can customize in `config.json`:

```json
{
  "network_type": "facebook",
  "network_file": "../data/facebook_combined.txt",
  "num_agents": 100,
  "high_rep_count": 3,
  "use_top_degree_influencers": true,
  ...
}
```

The simulation will:
1. Load the Facebook network from the edge list file
2. Sample `num_agents` nodes (if specified and smaller than total, selects top degree nodes)
3. Use community detection to assign groups
4. Select top `high_rep_count` degree nodes as influencers

**Note**: The full Facebook network has 4039 nodes. You can specify a smaller `num_agents` to sample a subset.

### Using Generated Network (Barabási-Albert)

To use a generated Barabási-Albert network:

1. Edit `config.json`:
```json
{
  "network_type": "barabasi_albert",
  "num_agents": 100,
  "high_rep_count": 3,
  "ba_m": 2,
  ...
}
```

2. Run the simulation:
```bash
python3 simulation.py
```

This will:
1. Generate a Barabási-Albert network with specified number of agents
2. Use first `high_rep_count` agents as influencers
3. Run simulation rounds
4. Generate network and metrics visualizations

## Modifying Parameters

### Option 1: Edit directly in code

Open `simulation.py` and modify the `config` dictionary in the `main()` function:

```python
config = {
    'num_agents': 20,           # Number of agents (start small!)
    'prob_in': 0.6,             # Probability of connection within group
    'prob_out': 0.1,            # Probability of connection between groups
    'num_groups': 2,            # Number of groups in network
    'message_left_bias': 2.0,   # Alpha parameter for message bias
    'message_right_bias': 2.0,  # Beta parameter for message bias
    'prob_truth': 0.5,          # Probability of message being true
    'agent_left_bias': 2.0,     # Alpha parameter for agent bias
    'agent_right_bias': 2.0,    # Beta parameter for agent bias
    'ave_reputation': 0.0,      # Mean of initial reputation
    'variance_reputation': 1.0,  # Variance of initial reputation
    'bias_strength': 0.3,       # Weight of ideological alignment (omega)
    'reputation_reward_strength': 0.5,  # Reputation gain for truth (gamma)
    'reputation_penalty_strength': 0.5,  # Penalty for misinformation (delta)
    'forwarding_cost': 0.1,     # Cost of forwarding (k)
    'num_rounds': 10,           # Number of rounds
    'initial_senders': [0],     # IDs of agents that receive initial message
    'seed': 42                  # Seed for reproducibility
}
```

### Option 2: Use JSON file

Edit `config.json` and run:

```python
from simulation import Simulation
sim = Simulation(config_file='config.json')
sim.run()
sim.visualize_network(round_result=sim.run_round(0))
sim.plot_metrics()
```

## Important Parameters

### Network
- **network_type**: Type of network to use - `'facebook'` (default) or `'barabasi_albert'`
- **network_file**: Path to Facebook edge list file (required if `network_type='facebook'`)
- **num_agents**: Number of agents in the network (works for both types)
  - For Facebook: if specified and smaller than total nodes, samples top degree nodes
  - For Barabási-Albert: exact number of agents to generate
- **high_rep_count**: Number of influencers (default: 3)
  - For Facebook: selects top degree nodes as influencers
  - For Barabási-Albert: uses first `high_rep_count` agents
- **ba_m**: Number of edges to attach from a new node to existing nodes in Barabási-Albert model (default: 2)
- **num_groups**: Number of groups for visualization purposes
  - For Barabási-Albert: groups assigned based on node degree
  - For Facebook: groups assigned using community detection (Louvain algorithm)
- **use_top_degree_influencers**: If `true`, selects top degree nodes as influencers (default: true, recommended for Facebook networks)
- **Note**: Parameters `prob_in` and `prob_out` are kept for compatibility but not used.

### Messages
- **prob_truth**: Probability of a message being true (0-1)
- **message_left_bias / message_right_bias**: Beta distribution parameters for message ideological bias

### Agents

**High Reputation Type (Influencers)**:
- **high_rep_count**: Number of influencers (2-4). These are the ones who start the news.
- **high_rep_reward (γ)**: High reputation gain when forwarding true messages (default: 1.0)
- **high_rep_penalty (δ)**: High penalty when forwarding false messages (default: 1.0)

**Low Reputation Type (Regular Users)**:
- **low_rep_reward (γ)**: Low reputation gain (default: 0.5)
- **low_rep_penalty (δ)**: Low penalty (default: 0.5)

**Common Parameters**:
- **bias_strength (ω)**: How much agents value ideological alignment
- **forwarding_cost (k)**: Fixed cost of forwarding a message

### Simulation
- **num_rounds**: Number of rounds to execute
- **Note**: The `initial_senders` are automatically the first `high_rep_count` agents (the influencers)

## Visualizations

The simulation generates three files:

1. **network_diffusion.png**: Shows the network structure and message diffusion in the last round
   - Larger blue nodes = Influencers (high_reputation)
   - Smaller nodes = Regular users (low_reputation)
   - Red = forwarded the message
   - Orange = received but didn't forward
   - Gray = didn't receive

2. **metrics.png**: Graphs of metrics over time
   - Message reach
   - Forwarding rate
   - Misinformation contamination
   - Average reputation

3. **diffusion_animation.gif**: Animated GIF showing diffusion in each round
   - Each frame shows one round of the simulation
   - Visualizes how the message spreads through the network over time
   - Colors indicate the state of each agent (received, forwarded, etc.)

## Code Structure

- `simulation.py`: Main script to run simulations
- `snlearn/simulation.py`: Simulation class implementation
- `snlearn/agent.py`: Agent implementation
- `snlearn/message.py`: Message implementation
- `snlearn/socialnetwork.py`: Social network generation and management
- `config.json`: Configuration file for Barabási-Albert network
- `config_facebook.json`: Example configuration file for Facebook network
- `data/facebook_combined.txt`: Facebook network edge list (if available)

## Using SocialNetwork Class Directly

The `SocialNetwork` class has a clean API with `num_agents` and `seed` as main parameters, and network-specific configurations in dictionaries:

### Barabási-Albert Network

```python
from snlearn.socialnetwork import SocialNetwork

# Create a Barabási-Albert network
network = SocialNetwork(
    num_agents=100,
    seed=42,
    barabasi_params={
        'm': 3,           # Number of edges to attach from new node
        'num_groups': 2   # Number of groups for visualization
    }
)
```

### Facebook Network

```python
from snlearn.socialnetwork import SocialNetwork

# Load Facebook network
network = SocialNetwork(
    num_agents=100,  # Sample 100 nodes (optional, uses all if not specified)
    seed=42,
    facebook_params={
        'network_file': '../data/facebook_combined.txt',
        'num_groups': 2   # Number of groups for visualization
    }
)
```

### Default (Barabási-Albert with defaults)

```python
from snlearn.socialnetwork import SocialNetwork

# Simple network with default parameters (m=2, num_groups=2)
network = SocialNetwork(num_agents=50, seed=42)
```

## Example: Small and Simple Network

To start with a very simple network:

```json
{
  "num_agents": 20,
  "num_groups": 2,
  "ba_m": 2,
  "high_rep_count": 2,
  "high_rep_reward": 1.0,
  "high_rep_penalty": 1.0,
  "low_rep_reward": 0.5,
  "low_rep_penalty": 0.5,
  "num_rounds": 5,
  "seed": 42
}
```

This creates a small Barabási-Albert network with 20 agents, 2 influencers who start the news, and runs only 5 rounds. The `ba_m` parameter controls how many connections each new node makes (higher values create denser networks).

## Modifying the Number of Influencers

To change how many influencers exist in the network, edit `high_rep_count` in `config.json`:

```json
"high_rep_count": 3  // Can be 2, 3, or 4
```

The first `high_rep_count` agents will be the influencers and will automatically start the messages.
