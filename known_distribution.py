import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy.stats import gaussian_kde

# Set random seed for reproducibility
np.random.seed(42)

# Generate 3 Gaussian blobs
n_samples = 1000
centers = [(1, 1), (-1, -1), (1, -1)]
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.5, random_state=42)

# Set manual image resolution (DPI and figure size)
dpi = 300
figsize = (8, 4)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

# Overplotted scatterplot
ax1.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5, c='blue')
ax1.set_title('Overplotted Scatterplot')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Density 2D plot
sns.kdeplot(x=X[:, 0], y=X[:, 1], cmap='viridis', fill=True, ax=ax2)
ax2.set_title('Density 2D Plot')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.show()