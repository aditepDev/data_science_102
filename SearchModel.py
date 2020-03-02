import math
import time
import numpy  as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import PathFile
from ReadData import readdata


def GridSearch(name_model,verbose):
    print("################## start ################## :: ",name_model)
    try:
        t0 = time.time()
        xxx = list(range(10000000))
        t1 = time.time()
        hidden_layer_List = [0]
        if name_model == 'mlpc':
            loop = np.arange(1, 10)
            layer_List = []
            hidden_layer_List = []
            for a in loop:
                for b in loop:
                    for c in loop:
                        layer_List = [a, b, c]
                        hidden_layer_List.append(layer_List)
                    layer_List = [a, b]
                    hidden_layer_List.append(layer_List)
                layer_List = [a, ]
                hidden_layer_List.append(layer_List)
        print(hidden_layer_List)
        param_grid = {'rfc': [{'n_estimators': [100, 200, 300, 400, 500, 600],
                               'criterion': ['gini', 'entropy']
                               }],
                      'mlpc': [{'learning_rate_init': [0.5, 0.1, 0.01, 0.001, 0.00001],
                                'max_iter': [100, 200, 300, 400, 500],
                                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                'solver': ['lbfgs', 'sgd', 'adam'],
                                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                'hidden_layer_sizes': hidden_layer_List
                                }],
                      'knn': [{'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                               }],
                      'svc': [{'kernel': ['linear', 'rbf', 'poly'],
                               'gamma': [0.1, 1, 10, 100],
                               'C' : [0.1, 1, 10, 100, 1000],
                               'degree' : [0, 1, 2, 3, 4, 5, 6]
                               }]
                      }

        Classifier = {'rfc': RandomForestClassifier(), 'mlpc': MLPClassifier(), 'knn': KNeighborsClassifier(),
                      'svc': SVC()}
        print('เวลาเริ่ม: %f' % (t1 - t0))
        print(time.time() - t0)
        X, z = readdata()
        # print(param_grid[name_model])
        clf = GridSearchCV(Classifier[name_model], param_grid[name_model], cv=3,
                           scoring='accuracy', verbose=verbose,n_jobs=-1,pre_dispatch='2*n_jobs')
        print(" :........!!: ", name_model)
        clf.fit(X, z.values.ravel())
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        t0 = time.time()
        yyy = [math.sin(x) for x in xxx]
        t1 = time.time()
        print('เวลาในการคำนวณ: %f' % (t1 - t0))
        best_params = str(clf.best_params_)
        path = PathFile.READFILE_PARAMETER + name_model + '_parameter.txt'
        with open(path, 'w') as data_file:
            data_file.write(best_params)
            print('write...!')
            data_file.close()
        print("################## end ##################:: ", name_model)
    except Exception as e:
        print(e)


# hidden_layer_List()
classifierName = ['rfc','mlpc','knn','svc']
# classifierName = ['rfc','knn']
# GridSearch('mlpc',0)
for model in classifierName:
    GridSearch(model,10)