import pickle
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from ReadData import readdata
import matplotlib.pyplot as plt

import PathFile


class Train:
    def __init__(self, X, z):
        self.X = X
        self.z = z

    def train(self):
        classifiers = [
            ["Rfc", Rfc(criterion="entropy", n_estimators=100)],
            ["knn", Knn(10, algorithm="auto")],
            ["svc", SVC(kernel="linear", C=0.025, verbose=True)],
            ["MLPC",
             MLPC(activation='identity', learning_rate_init=0.01, hidden_layer_sizes=(3, 2, 2),
                  learning_rate='adaptive',
                  solver='adam', verbose=True, max_iter=100)]
        ]

        def dump_Data(fileName, model):
            try:
                f = open(PathFile.PREDICRFILE + fileName + ".pkl", "wb")
                pickle.dump(model, f)
                f.close()
                print(fileName, 'Dump_file OK...')
            except IOError as e:
                print(e)

        for name, model in classifiers:
            model.fit(self.X, self.z.values.ravel())
            dump_Data(name, model)
        return model

    def showData(self):
        z = self.z.produce.to_numpy()
        X = KMeans(2).fit_transform(self.X)
        plt.figure(figsize=[7, 7]).gca(aspect=1)
        plt.gca(aspect=1).scatter(X[:, 0], X[:, 1], c=z, edgecolor='k', cmap='jet')
        plt.show()


if __name__ == '__main__':
    X, z = readdata()
    training = Train(X, z);
    # training.train()
    training.showData()
