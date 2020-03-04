import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, DBSCAN, OPTICS, AffinityPropagation
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import PathFile
import sys
import PathFile

def showplt(title, X, z, model, filename):
    # if title == 'KMeans' or title == 'MeanShift' or title == 'AffinityPropagation':
    #     if title == 'KMeans':
    #         X = model.transform(X)
    #     plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], 300, '#AAEE55', marker='*',
    #                 edgecolor='#DD9900',
    #                 lw=2)
    X = model.transform(X)
    unique_elements, counts_elements = np.unique(z, return_counts=True)
    a,b = np.asarray((unique_elements, counts_elements))
    cb = [u'กลุ่ม %d\n จำนวน %d' % (c, b) for (c, b) in zip(a+1, b)]
    cmap = plt.cm.get_cmap('jet')
    plt.figure(figsize=[10, 10])
    for i in range(9):
        plt.gca(aspect=1).scatter(None, None, c=cmap(i/9), edgecolor='k')
        plt.gca(aspect=1).legend(cb, prop={'family': 'Tahoma'})
    plt.gca(aspect=1).scatter(X[:, 0], X[:, 1], c=z, edgecolor='k', cmap=cmap)
    plt.title(title + filename)
    plt.savefig(PathFile.READFILE_IMAGE + title + filename + '_2D.png')
    plt.show()


    if title == 'KMeans':
        plt.figure(figsize=[10, 10])
        ax = plt.axes([0, 0, 1, 1], projection='3d')
        for i in range(9):
            ax.scatter(None, None, c=cmap(i / 9), edgecolor='k')
            ax.legend(cb, prop={'family': 'Tahoma'})
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=z, edgecolor='k', cmap=cmap)
        # cen = model.transform(model.cluster_centers_)
        # ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], s=300, c='#AAEE55', marker='*', edgecolor='#DD9900', lw=2)
        plt.title(title + filename)
        plt.savefig(PathFile.READFILE_IMAGE +title + filename+'_3D.png')
        plt.show()


def dumpfile(z, c, n):
    # print("################## dumpfile_Start ##################:: ", c)
    path = PathFile.READFILE_CLUSTER + c + n + '_parameter.txt'
    with open(path, 'w') as data_file:
        print(z.size)
        data_file.write(str(z))
        print('write...!')
        data_file.close()
    # print("################## dumpfile_end ##################:: ", c)


def loadfile(c):
    # print("################## loadfile_Start ##################:: ", c)
    path = PathFile.READFILE_CLUSTER + c + '_parameter.txt'
    with open(path, 'rt') as data_file:
        print('loadfile...!')
        data = data_file.read()
    # print("################## loadfile_end ##################:: ", c)
    return data


def readdata(filename):
    print(PathFile.READFILE_EXCEL)
    data = pd.read_csv(PathFile.READFILE_EXCEL + filename + '.csv')
    data.info()
    X = pd.DataFrame(data)
    X = X.to_numpy()
    return X


def Dataset():
    np.random.seed(10)
    X, _ = datasets.make_blobs(n_samples=1500, n_features=2, centers=3, cluster_std=2.1)
    # dumpfile(X,'fileSet')
    # data = loadfile('fileSet')
    # print(data)
    return X


np.set_printoptions(threshold=sys.maxsize)
## clustering_algorithms
MeanShift = MeanShift(bandwidth=2)
KMeans = KMeans(9, verbose=1000)
SpectralClustering = SpectralClustering(n_clusters=9, assign_labels="discretize", random_state=0)
DBSCAN = DBSCAN(eps=3, min_samples=2)
OPTICS = OPTICS(min_samples=2)
AffinityPropagation = AffinityPropagation()
# clustering_algorithms  = [['MeanShift', MeanShift], ['KMeans', KMeans],
#               ['SpectralClustering', SpectralClustering], ['DBSCAN', DBSCAN], ['OPTICS', OPTICS],['AffinityPropagation',AffinityPropagation]]
file = [['1k','0-0-1000'],['10k','0-0-10000'],['100k','0-0-100k'],['1m','0-0-1m']]
clustering_algorithms = [['KMeans', KMeans]]
# file = [['1k','0-0-1000']]
for n, f in file:
    X = readdata(f)
    for c, model in clustering_algorithms:
        print(c, "Start.......!")
        z = model.fit_predict(X)
        print(c, "predict.......OK!")
        showplt(title=c, z=z, X=X, model=model, filename=n)
        print(c, "showplt.......OK!")
        dumpfile(z, c, n)
        print(c, "dumpfile.......OK!")
        print(c, "######## END ########")
