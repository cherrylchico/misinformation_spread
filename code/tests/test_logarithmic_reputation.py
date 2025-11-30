"""
Test script to demonstrate logarithmic reputation gains
Shows how reputation gains decrease as reputation increases
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from snlearn.agent import Agent
from snlearn.message import Message

# Create an agent with standard parameters
agent = Agent(
    left_bias=2.0,
    right_bias=2.0,
    ave_reputation=0.5,
    variance_reputation=0.1,
    bias_strength=0.5,
    reputation_reward_strength=0.25,  # Base reward strength
    reputation_penalty_strength=0.25,
    forwarding_cost=0.1
)

# Create a truthful message
message = Message(
    left_bias=0.5,
    right_bias=0.5,
    prob_truth=1.0  # Always truthful
)
message.truth_revealed = True

# Test reputation gains at different reputation levels
initial_reputations = np.linspace(-0.5, 3.0, 20)
gains = []

for initial_rep in initial_reputations:
    # Set agent's reputation
    agent.reputation = initial_rep
    agent.current_action = 1  # Agent forwards the message
    
    # Store reputation before update
    rep_before = agent.reputation
    
    # Update reputation
    agent.update_reputation(message, store=False)
    
    # Calculate gain
    gain = agent.reputation - rep_before
    gains.append(gain)
    
    # Reset for next iteration
    agent.reputation = initial_rep

# Create visualization
plt.figure(figsize=(12, 5))

# Plot 1: Reputation gain vs initial reputation
plt.subplot(1, 2, 1)
plt.plot(initial_reputations, gains, 'b-', linewidth=2, label='Logarithmic gains')
plt.axhline(y=0.25, color='r', linestyle='--', label='Linear gain (baseline)')
plt.xlabel('Initial Reputation', fontsize=12)
plt.ylabel('Reputation Gain', fontsize=12)
plt.title('Logarithmic Reputation Gains\n(Diminishing Returns)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Gain ratio (compared to linear)
plt.subplot(1, 2, 2)
gain_ratios = [g / 0.25 for g in gains]
plt.plot(initial_reputations, gain_ratios, 'g-', linewidth=2)
plt.axhline(y=1.0, color='r', linestyle='--', label='Linear baseline')
plt.xlabel('Initial Reputation', fontsize=12)
plt.ylabel('Gain Ratio (vs Linear)', fontsize=12)
plt.title('Relative Gain Reduction\n(Logarithmic vs Linear)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

# Save to output directory with timestamp
from snlearn.output_manager import OutputManager
output_path = OutputManager.get_output_path('logarithmic_reputation_test.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Print some example values
print("\n" + "="*60)
print("LOGARITHMIC REPUTATION GAINS - TEST RESULTS")
print("="*60)
print(f"Base reward strength: {agent.reputation_reward_strength}")
print("\nGains at different reputation levels:")
print("-"*60)
for i, (rep, gain) in enumerate(zip(initial_reputations, gains)):
    if i % 3 == 0:  # Print every 3rd value to avoid clutter
        reduction = (1 - gain/0.25) * 100
        print(f"Initial Rep: {rep:6.2f} → Gain: {gain:.4f} (Reduction: {reduction:5.1f}%)")
print("="*60)

plt.show()
