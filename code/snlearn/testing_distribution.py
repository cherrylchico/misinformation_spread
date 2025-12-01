import numpy as np
import matplotlib.pyplot as plt

# Your class-style logic replicated as a function for plotting
def sample_bias(left_bias, right_bias, size=10_000):
    # Draw from Beta
    samples = np.random.beta(left_bias, right_bias, size=size)
    # Convert [0,1] → [-1,1]
    bias = 2 * samples - 1
    return bias

# Choose the beta parameters
left_bias = 2
right_bias = 2

# Generate data
bias_values = sample_bias(left_bias, right_bias)

# Plot
plt.figure(figsize=(8, 5))
plt.hist(bias_values, bins=50, density=True, alpha=0.7)
plt.title(f"Distribution of Bias (Beta({left_bias}, {right_bias}) mapped to [-1,1])")
plt.xlabel("bias value")
plt.ylabel("density")
plt.grid(True, alpha=0.3)
plt.show()
