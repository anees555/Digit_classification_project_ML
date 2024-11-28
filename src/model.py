import os
from sklearn.svm import SVC
from  sklearn.metrics import accuracy_score
from data_preprocess import data_preprocessing
from joblib import dump

def train_model(X_train, y_train):
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok  = True)
    model = SVC(kernel = 'linear')
    model.fit(X_train, y_train)
    dump(model, f"{save_dir}/digit_classifier_model.joblib") 
    print("file  loades successfully")
    return model


print("modelling")


