# Misinformation Diffusion Simulation

This code implements the misinformation diffusion model in social networks with interactive visualizations and advanced network sampling methods.

## Features

- **Multiple Network Types**:
  - Facebook network (real-world social network data)
  - Barabási-Albert generated networks
  
- **Advanced Sampling Methods** (for Facebook networks):
  - **Community-based sampling**: Preserves community diversity with stratified sampling (high, medium, and low-degree nodes from each community)
  - **Degree-based sampling**: Selects top-degree nodes (hub-biased)
  - **Default sampling**: Uses degree-based for backward compatibility

- **Two Types of Agents**:
  - **Influencers** (2-4 agents): High reputation sensitivity, start spreading messages
  - **Regular users**: Lower reputation sensitivity, majority of network

- **Visualizations**:
  - Network structure with community detection
  - Animated GIF showing diffusion round-by-round
  - Metrics over time (reach, forwarding rate, contamination)

## Installation

```bash
# Navigate to code directory
cd code

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Main Simulation

Run the simulation with configuration from `config.json`:

```bash
cd code
python simulation.py
```

Output files will be saved in `figures/`:
- `network_diffusion.png` - Network structure and final diffusion state
- `metrics.png` - Time-series metrics
- `diffusion_animation.gif` - Animated diffusion process

### 2. Visualize Network Structure

Generate network visualizations with different sampling methods:

```bash
cd code

# Facebook network with community-based sampling (recommended)
python visualize_network.py 100 facebook --sampling_method community

# Facebook network with degree-based sampling
python visualize_network.py 100 facebook --sampling_method degree

# Facebook network without explicit sampling (uses default)
python visualize_network.py 100 facebook

# Barabási-Albert network
python visualize_network.py 100 barabasi --m 3
```

**Parameters:**
- First argument: Number of agents (required)
- Second argument: Network type - `facebook` or `barabasi` (required)
- `--sampling_method`: For Facebook networks - `community` or `degree` (optional)
- `--m`: Edges per node for Barabási-Albert (default: 2)
- `--num_groups`: Communities to visualize (default: 5)
- `--seed`: Random seed (default: 42)

**Examples:**
```bash
# Custom parameters
python visualize_network.py 200 facebook --sampling_method community --num_groups 8 --seed 123

# Barabási-Albert with custom m parameter
python visualize_network.py 150 barabasi --m 4 --num_groups 3
```

Output saved to `figures/` with filename format: `{network_type}_{sampling_method}_{num_agents}nodes.png`

## Configuration

### Using config.json

Edit `code/config.json` to customize simulation parameters:

```json
{
  "network_type": "facebook",
  "network_file": "../data/facebook_combined.txt",
  "num_agents": 100,
  "sampling_method": "community",
  "num_groups": 5,
  "high_rep_count": 3,
  "high_rep_reward": 1.0,
  "high_rep_penalty": 1.0,
  "low_rep_reward": 0.5,
  "low_rep_penalty": 0.5,
  "bias_strength": 0.3,
  "forwarding_cost": 0.1,
  "num_rounds": 10,
  "seed": 42
}
```

### Using SocialNetwork API Directly

```python
from snlearn.socialnetwork import SocialNetwork

# Facebook with community-based sampling
network = SocialNetwork(
    num_agents=100,
    seed=42,
    facebook_params={
        'network_file': 'data/facebook_combined.txt',
        'num_groups': 5,
        'sampling_method': 'community'  # 'community', 'degree', or None
    }
)

# Barabási-Albert network
network = SocialNetwork(
    num_agents=100,
    seed=42,
    barabasi_params={
        'm': 3,           # Edges to attach from new node
        'num_groups': 5
    }
)
```

## Key Parameters

### Network Configuration
- **network_type**: `'facebook'` or `'barabasi_albert'`
- **num_agents**: Number of agents in the network
- **sampling_method**: For Facebook networks:
  - `'community'`: Stratified sampling across communities (recommended for research)
  - `'degree'`: Top-degree nodes only (hub-biased)
  - `None`: Default degree-based sampling
- **num_groups**: Number of communities for visualization

### Sampling Method Comparison

**Community-Based Sampling** (Recommended):
- ✅ Preserves community diversity
- ✅ Stratified: 40% high-degree, 30% medium, 30% low-degree nodes from each community
- ✅ Includes bridge nodes for connectivity
- ✅ Better representativeness for misinformation research
- Complexity: O(N + E) for community detection

**Degree-Based Sampling**:
- ❌ Biased toward hubs (influencers only)
- ❌ Poor community diversity
- ✅ Fast and simple: O(N log N)
- ✅ Always connected
- Use when: studying hub behavior specifically

### Agent Types

**Influencers (high_rep_count agents)**:
- `high_rep_reward` (γ): Reputation gain for forwarding truth (default: 1.0)
- `high_rep_penalty` (δ): Penalty for forwarding misinformation (default: 1.0)
- Start spreading messages

**Regular Users**:
- `low_rep_reward`: Lower reputation gain (default: 0.5)
- `low_rep_penalty`: Lower penalty (default: 0.5)

### Message & Behavior
- `prob_truth`: Probability message is true (0-1)
- `bias_strength` (ω): Weight of ideological alignment
- `forwarding_cost` (k): Cost of forwarding
- `num_rounds`: Simulation rounds

## Output Files

All visualizations are saved to `figures/`:

1. **Network Visualizations** (from `visualize_network.py`):
   - Node size = degree (connections)
   - Node color = community
   - Shows community structure and degree distribution
   - Filename: `{network}_{method}_{agents}nodes.png`

2. **Simulation Outputs** (from `simulation.py`):
   - `network_diffusion.png`: Final diffusion state
   - `metrics.png`: Time-series graphs
   - `diffusion_animation.gif`: Round-by-round animation

## Project Structure

```
.
├── code/
│   ├── simulation.py              # Main simulation script
│   ├── visualize_network.py       # Network visualization tool
│   ├── config.json                # Simulation configuration
│   ├── requirements.txt           # Python dependencies
│   ├── snlearn/                   # Core package
│   │   ├── simulation.py          # Simulation class
│   │   ├── socialnetwork.py       # Network generation & sampling
│   │   ├── agent.py               # Agent implementation
│   │   └── message.py             # Message implementation
│   └── README.md                  # Detailed code documentation
├── data/
│   └── facebook_combined.txt      # Facebook network edge list
├── figures/                       # Output visualizations
└── paper/                         # LaTeX paper files
```

## Examples

### Example 1: Compare Sampling Methods

```bash
cd code

# Generate community-based sample
python visualize_network.py 100 facebook --sampling_method community

# Generate degree-based sample
python visualize_network.py 100 facebook --sampling_method degree

# Compare the resulting PNG files in figures/
```

### Example 2: Run Full Simulation with Community Sampling

Edit `code/config.json`:
```json
{
  "network_type": "facebook",
  "sampling_method": "community",
  "num_agents": 100,
  "num_rounds": 10
}
```

Then run:
```bash
cd code
python simulation.py
```

### Example 3: Small Test Network

```bash
cd code
python visualize_network.py 50 barabasi --m 2 --num_groups 3
```

## Research Notes

### Why Community-Based Sampling?

For misinformation research, community-based sampling provides:
1. **Diverse social contexts**: Captures different communities with varying connectivity
2. **Realistic diffusion patterns**: Mix of well-connected and peripheral nodes
3. **Bridge nodes**: High-degree nodes that connect communities
4. **Structural preservation**: Maintains degree distribution within communities

Degree-based sampling only captures hubs, which biases results toward influencer behavior and misses community-level dynamics.

### Visualization Interpretation

In network visualizations:
- **Node size** = Degree (number of connections)
- **Node color** = Community membership
- **Larger nodes** = Hubs/bridges (high connectivity)
- **Smaller nodes** = Peripheral members (lower connectivity)

Community-based samples show varied node sizes within each color (community), while degree-based samples show mostly large nodes (all hubs).

## Citation

If you use this code for research, please cite the associated paper.

## License

[Add your license information here]
