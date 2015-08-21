__author__ = 'dario'


import numpy as np
import matplotlib.pyplot as plt

# Generate some test data
x = np.random.randn(8873)
y = np.random.randn(8873)

# This makes a 50x50 heatmap. If you want, say, 512x384, you can put bins=(512, 384) in the call to histogram2d
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap, extent=extent)
plt.show()