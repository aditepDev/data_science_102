
import matplotlib.pyplot as plt
import numpy  as  np
from sklearn import datasets
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from TrainModel import Train
from PredictData import predictData
from ReadData import readdata

def showData(X,z):
    cmap = plt.cm.get_cmap('jet')
    z = z.produce.to_numpy()
    X = KMeans(2).fit_transform(X)
    plt.figure(figsize=[7,7]).gca(aspect=1)
    plt.gca(aspect=1).scatter(X[:, 0], X[:, 1], c=z, edgecolor='k', cmap=cmap)
    plt.show()



def test(X):
    # z = [[2563, 18, 7, 32, 10, 8, 2000, 12, 101], [2563, 23, 5, 32, 10, 7, 1000, 13, 101],
    #      [2563, 1, 2, 32, 10, 9, 190, 16, 101]]
    for t in X:
        print('ปี :[', t[0], ']  จำนวนแหล่งน้ำในแปลงใหญ่ : [', t[1], '] ภัยพิบัติ : [', t[2], '] รหัสจังหวัด : [', t[3],
              '] รหัสอำเภอ : [', t[4], '] รหัสตำบล : [', t[5], '] ค่าบำรุ่งรักษา : [', t[6], '] ราคาผลผลิต : [', t[7],
              '] รหัสสายพันธุ์ : [', t[8], ']')
        predictData(t)

if __name__ == '__main__':
    X, z = readdata()
    showData(X,z)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    Train(X, z)
    # test(X_test)



