from scipy.cluster.vq import kmeans2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, DBSCAN, OPTICS, AffinityPropagation
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import PathFile
import sys


n_clusters = 10
df = pd.DataFrame({'x':np.random.randn(1000), 'y':np.random.randn(1000)})
_, df['cluster'] = kmeans2(df, n_clusters)

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('jet')
for i, cluster in df.groupby('cluster'):
    print(i)
    # print(cluster)
    print(i/n_clusters)
    # _ = ax.scatter(cluster['x'], cluster['y'], c=cmap(i / n_clusters), label=i)
ax.legend()