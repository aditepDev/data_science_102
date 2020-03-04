import pickle
import numpy as np
import PathFile

def predictData(data):
    t = np.array([data])

    # fileNames = ["knn", "svc", "MLPC","Rfc"]
    fileNames = ["MLPC"]
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

