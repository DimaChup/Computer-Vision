# The following code creates some artificial data and plots it
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
import numpy as np

X, y = make_blobs(centers=3, n_samples=100, cluster_std=2, random_state=42)
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1]);

plt.show()
