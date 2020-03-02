import pandas as pd
import PathFile
import numpy as np


def correcting(X):
    for d in X:
        xd = np.mean(X[d])
        X[d] = X[d].fillna(xd)
    return X


def readdata():
    print(PathFile.READFILE_DATA)
    data = pd.read_csv(PathFile.READFILE_DATA)
    data.info()
    datas = ['year', 'water', 'disaster', 'province', 'amphur', 'tambon', 'maintenance', 'price', 'breed', 'produce']
    X = pd.DataFrame(data, columns=datas)
    # X = X[(X['year'] > 2000) & (X['year'] < 3000) & (X['breed'] == 302)]
    X = correcting(X)
    X.to_csv(PathFile.READFILE_EXCEL + 'dataframe.csv')
    z = pd.DataFrame(X, columns=['produce'])
    X = pd.DataFrame(X, columns=['year', 'water', 'disaster', 'province', 'amphur', 'tambon', 'maintenance', 'price',
                                 'breed'])
    z = z.astype('int')
    X = X.to_numpy()
    return X, z


# testdate()
# #
# X, z = readdata()
# print(X)
# cleardata()