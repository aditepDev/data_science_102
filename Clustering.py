import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, SpectralClustering,DBSCAN,OPTICS
from sklearn import datasets

import PathFile


def showplt(title, X, z):
    plt.figure(figsize=[7, 7])
    plt.gca(aspect=1).scatter(X[:, 0], X[:, 1], c=z, edgecolor='k', cmap='jet')
    plt.title(title)
    plt.show()

def dumpfile(z,c):
    path = PathFile.READFILE_CLUSTER + c + '_parameter.txt'
    with open(path, 'w') as data_file:
        data_file.write(str(z))
        print('write...!')
        data_file.close()
    print("################## end ##################:: ", c)


np.random.seed(10)
X, _ = datasets.make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=9)
clustering = [['MeanShift', MeanShift(bandwidth=2)], ['KMeans', KMeans(9)],
              ['SpectralClustering', SpectralClustering(n_clusters=9, assign_labels="discretize", random_state=0)],['DBSCAN',DBSCAN(eps=3, min_samples=2)],['OPTICS', OPTICS(min_samples=2)]]
for c, model in clustering:
    print(c, "Start.......!")
    z = model.fit_predict(X)
    showplt(title=c, z=z, X=X)
    dumpfile(z,c)
    print(c, "######## END ########")

