# MachineLearning
### Classification 

` sklearn.ensemble.RandomForestClassifier`
```python
RandomForestClassifier(n_estimators=200)
n_estimators, optional (default=100) # จำนวนต้นไม้
criterion, optional (default=”gini”) (gini/entropy)
```
`sklearn.neighbors.KNeighborsClassifier`
```python
n_neighborsint, optional (default = 5) # จำนวนของเพื่อนบ้านที่จะใช้เป็นค่าเริ่มต้น
weights : str or callable, optional (default = ‘uniform’) # ฟังก์ชั่นน้ำหนักที่ใช้ำนายค่าที่เป็นไปได้
‘uniform’ : ตุ้มน้ำหนักที่สม่ำเสมอ คะแนนทั้งหมดในแต่ละย่านมีน้ำหนักเท่ากัน
‘distance’ : คะแนนน้ำหนักโดยการผกผันของระยะทางของพวกเขา ในกรณีนี้เพื่อนบ้านที่อยู่ใกล้จุดสอบถามจะมีอิทธิพลมากกว่าเพื่อนบ้านที่อยู่ไกลออกไป
```

### train Model
```python
X, z = readdata()
model = train(X, z)

def train(X, z):
    classifiers = [
        ["Rfc",Rfc(n_estimators=200)]
        ["knn", Knn(3)],
        ["svc", SVC(kernel="linear", C=0.025 ,verbose=True)],
        ["MLPC", MLPC(activation = 'tanh',learning_rate_init=0.0001, hidden_layer_sizes = (100,100,100), learning_rate = 'adaptive' ,solver = 'adam' ,verbose=True ,max_iter=1000)]
    ]

    def dump_Data(fileName, model):
        try:
            f = open(PathFile.PREDICRFILE + fileName + ".pkl", "wb")
            pickle.dump(model, f)
            f.close()
            print(fileName,'Dump_file OK...')
        except IOError as e:
            print(e)

    for name, model in classifiers:
        model.fit(X, z.values.ravel())
        dump_Data(name, model)
    return model

```
### PredictData
```python

  z = [[2563, 18, 0, 32, 10, 8, 1000, 13, 101],[2563, 18, 0, 32, 10, 8, 1000, 800, 101],[2563, 18, 8, 32, 10, 8, 1000, 100, 205]]
    for t in z:
        print('ปี :[',t[0],']  จำนวนแหล่งน้ำในแปลงใหญ่ : [',t[1],'] ภัยพิบัติ : [',t[2],'] รหัสจังหวัด : [',t[3],'] รหัสอำเภอ : [',t[4],'] รหัสตำบล : [',t[5],'] ค่าบำรุ่งรักษา : [',t[6],'] ราคาผลผลิต : [',t[7],'] รหัสสายพันธุ์ : [',t[8],']')
        predictData(t)


def predictData(data):
    t = np.array([data])

    fileNames = ["knn", "svc", "MLPC","Rfc"]
    def load_Data(fileName):
        try:
            file_model = open(PathFile.PREDICRFILE + fileName + ".pkl", "rb")
            model = pickle.load(file_model)
            file_model.close()
        except IOError as e:
            print(e)
        return model

    for name in fileNames:
        model = load_Data(name)
        answer = model.predict(t)
        print("คำตอบที่เทำนาย : ", name, answer)

```
### Showdata
```python
def showData(X,z):
    z = z.produce.to_numpy()
    X2 = KMeans(2).fit_transform(X)
    plt.figure().gca(aspect=1)
    plt.scatter(X2[:, 0], X2[:, 1], c=z, edgecolor='k', cmap='rainbow')
    plt.show()

```
### แบ่งข้อมูลใช้ทดสอบ
```python
X = [[1,1],[1,0],[0,1],[0,0]]
z = [1,0,0,0]
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

classifiers = [
        ["knn", Knn(3)],
        ["svc", SVC(kernel="linear", C=0.025)],
        ["MLPC", MLPC(activation = 'identity', hidden_layer_sizes = (3, 3, 3, 3, 3, 3), learning_rate = 'invscaling' ,solver = 'adam')]
    ]
     for name, model in classifiers:
         model.fit(X_train, z_train)
         answer = model.predict(X_test)
         print(classification_report(z_test, answer))
```
### Dump File Model
```python
def dump_Data(fileName, model):
try:
    f = open(PathFile.PREDICRFILE + fileName + ".pkl", "wb")
    pickle.dump(model, f)
    f.close()
    print('Dump_file OK...')
except IOError as e:
    print(e)
```
### Load File Model
```python
fileNames = ["knn", "svc", "MLPC"]
def load_Data(fileName):
try:
    file_model = open(PathFile.PREDICRFILE + fileName + ".pkl", "rb")
    model = pickle.load(file_model)
    file_model.close()
except IOError as e:
    print(e)
return model
```
### โหลดข้อมูลจาก DB
```python
   if platform == "win32":
            mariadb_connection = mariadb.connect(host="localhost", user='root', password='1234', database='jobs')
        else:
            mariadb_connection = mariadb.connect(host="10.3.33.187", user='root', password='d0aep@ssw0rd',
                                                 database='test')
        cursor = mariadb_connection.cursor()
        cursor.execute("SELECT * FROM scheduled_bigfarm_data_temp")
        data = pd.DataFrame(cursor)
        X = pd.DataFrame(data,columns=[2,3,4,5,6,7,8,9,10,11,12,13])
        X = X[(X[3].astype('int') > 2000) & (X[3].astype('int') < 2600) & (X[9] < 100) & (X[10] < 1000)]
        X = correcting(X)
        X.info()
        X.to_csv(PathFile.READFILE_EXCEL + 'dataframeDB.csv')
        z = pd.DataFrame(X, columns=[10])
        X = pd.DataFrame(X,columns=[2,3,4,5,6,7,8,9,11,12,13])
        z = z.astype('int')
        X = X.to_numpy().astype('float')
        return  X,z
```
### โหลดข้อมูลจาก  excel
```python
# path
import os
script_dir = os.path.dirname(__file__)
READFILE = script_dir + "//Storage//Excel//part3new-rice.xlsx"

# โหลด
data = pd.read_excel(PathFile.READFILE)
```
```
def readdata():
    print(PathFile.READFILE)
    data = pd.read_excel(PathFile.READFILE)

    datas = ['year','water','disaster', 'province', 'amphur', 'tambon', 'maintenance','price','breed','produce']
    X = pd.DataFrame(data, columns=datas)
    X = X[(X['year'] > 2000) & (X['year'] < 3000)]
    for d in datas:
        xd = np.mean(X[d])
        X[d] = X[d].fillna(xd)

    X.to_csv(PathFile.script_dir + '/Storage/Excel/dataframe.csv')
    z = pd.DataFrame(X, columns=['produce'])
    X = pd.DataFrame(X, columns=['year','water','disaster', 'province', 'amphur', 'tambon', 'maintenance','price','breed'])
    z = z.astype('int')
    X = X.to_numpy()
    return X, z
```
### โหลดหลายไฟล์แล้วนำมารวมกัน
```python
 filename = ['2553','2554','2555','2556','2557','2558','2559','2560','2561','2562']
        datafram = []
        for name in filename:
            print(PathFile.READFILE_EXCEL+ 'rice1-'+name+'.xlsx')
            data = pd.read_excel(PathFile.READFILE_EXCEL+ 'rice1-'+name+'.xlsx')
            data = pd.DataFrame(data,columns=[2,6,7,8,9])
            X = data[(data[2] > 99999) & (data[6] != '-') & (data[7] != '-') & (data[8] != '-') & (data[8] != 0 ) & (data[9] != '-')]
            datafram.append(X)
            # X.info()
        datalist = []
        for i in range(len(datafram)):
            datalist.append(datafram[i])
        X = pd.concat(datalist)
        X.to_csv(PathFile.READFILE_EXCEL + 'dataframeDoae.csv')

        z = pd.DataFrame(X, columns=[8])
        X = pd.DataFrame(X, columns=[2,6,7,9])
        z = z.astype('int')
        X = X.to_numpy()

```

### เติมค่าเฉลี่ยเข้าช่องที่ว่าง วิธีแทนค่า Missing Value ด้วยค่าเฉลี่ย (Mean Imputation)
```python

datas = ['F311', 'year', 'F31B', 'F121B', 'F121A']
X = pd.DataFrame(data, columns=datas)

for d in datas:
   xd = np.mean(X[d])
   X[d] =  X[d] .fillna(xd)
       
F311 = np.mean(X.F311)
X.F311 = X.F31B.fillna(F311)

```

### แปลงข้อมูล DataFrame เป็น numpy array
`  X = pd.DataFrame(data, columns= ['idplang','year','F31B','F121B','F121A','allyear']).to_numpy() `

### โหลดข้อมูลจาก database
```python
import mysql.connector as mariadb

mariadb_connection = mariadb.connect(host="localhost", user='root', password='1234', database='zyanwoadev')
cursor = mariadb_connection.cursor()
cow_id = 1

cursor.execute("select * from tbd_cow WHERE cow_id=%s", (cow_id,))
for data in cursor:
    model = TbdCowModel(data[0])
    print(model.getCowID())
```


### ทดลอง TrainModel GridSearchCV
```python
import math
import time
import numpy  as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from ReadData import readdata
try:
    t0 = time.time()
    xxx = list(range(10000000))
    t1 = time.time()
    print('เวลาเริ่ม: %f' % (t1 - t0))
    print(time.time() - t0)
    X, z = readdata()
    loop = np.arange(1,5)

    layer_List = []
    hidden_layer_List = []
    for a in loop:
        for b in loop:
            for c in loop:
                for d in loop:
                    for e in loop:
                        print(a, b, c, d,e)
                        layer_List = [a, b, c, d,e]
                    layer_List = [a, b, c, d]
                    hidden_layer_List.append(layer_List)
                layer_List = [a, b, c]
                hidden_layer_List.append(layer_List)
            layer_List = [a, b]
            hidden_layer_List.append(layer_List)
        layer_List = [a, ]
        hidden_layer_List.append(layer_List)
    param_grid = [
        {   'learning_rate_init' : [0.5,0.1,0.01,0.001,0.00001],
            'max_iter' : [100,200,300,400,500],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'hidden_layer_sizes': [1]
        }]

    print(param_grid)
    clf = GridSearchCV(MLPClassifier(), param_grid, cv=3,
                       scoring='accuracy', verbose=10)
    clf.fit(X, z.values.ravel())
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    t0 = time.time()
    yyy = [math.sin(x) for x in xxx]
    t1 = time.time()
    print('เวลาในการคำนวณ: %f' % (t1 - t0))
except Exception as e:
    print(e)
```
### Path File
```python

import os
from sys import platform
try:
    # Windows
    if platform == "win32":
        script_dir = os.path.dirname(__file__)
        PREDICRFILE = script_dir + "//Storage//Model//"
    else:
        READFILE_EXCEL = "Storage/Excel/"
        READFILE_EXCEL_DATA = "Storage/Excel/data/"

except ImportError as e:
    print('Error:')
    raise e
```
### รับพารามิเตอร์จาก console
```python
argumentList = sys.argv
    predictData(list(map(float, argumentList[1:12])))
```
# Clean Data 
1. Parsing คือ การแจกแจงข้อมูล หรือการใช้หัวข้อของชุดข้อมูล
2. Correcting คือ การแก้ไขข้อมูลที่ผิดพลาด
` หาค่าเฉลี่ย ค่าเบี่ยงเบียนมาตรฐาน หรือ standard deviation หรือ Clustering algorithm  `
3. Standardizing คือ การทำข้อมูลให้เป็นรูปแบบเดียวกัน
` Standard Normal Distribution เรียงข้อมูลให้อยู่ในรูป Normalization หรือ ระฆังคว่ำ` 
4. Duplicate Elimination คือ การลบชุดข้อความซ้ำซ้อนทิ้ง

## การเตรียมข้อมูลสำหรับ
### ข้อมูลต้องแยกความแตกต่างของแต่ละแถวได้ (กำจัดคอลัมน์ที่มีข้อมูลไม่ซ้ำกันเลยออก)
### ไม่ใช้ข้อมูลที่ไม่ซ้ำกันเลย  เพราะไม่มีความสัมพันธ์กันของข้อมูล
### ปรับข้อมูลให้สมดุล
       ` school = 0   # ม.6 `
       ` school = 1   # เทียบเข้า `
### ลดการกระจ่ายของข้อมูล (จัดกลุ่ม)
  ` {A, B+, B} เป็น High, เกรด{C+, C} เป็น Medium และ เกรด {D+, D, F, W, I} เป็น Low  `
       

