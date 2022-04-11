import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
sns.set(rc={'figure.figsize':(15,6)});


def standardize(data):
    ###standardize the dataset 
    ###cite : https://stackoverflow.com/questions/24645153/pandas-dataframe-columns-scaling-with-sklearn and  https://www.kaggle.com/viswanathanc/auto-mpg-linear-regression
    scaled_cols = data.columns
    minmax = MinMaxScaler()
    data[scaled_cols] = minmax.fit_transform(data[scaled_cols])
    return data


def split(data,target):
    ###split dataset into 60-20-20
    X_dev, X_test, y_dev, y_test = train_test_split(data, target, test_size=0.1, shuffle = True, random_state= 5) #test
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.1, shuffle = True, random_state = 5) #val
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_val shape:", X_val.shape)
    return X_dev, y_dev, X_test, y_test, X_train, y_train, X_val, y_val
    
    
def cm(y_pred, y_true, class_labels, model_name = ''):
    results = classification_report(y_pred, y_true, output_dict = True )
    macro_f1 = results['macro avg']['f1-score']
    accuracy = results['accuracy']
    #plotconfusion_matrix
    cm = confusion_matrix(y_pred, y_true)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = class_labels)
    disp.plot(xticks_rotation = 45, cmap=plt.cm.Blues)
    plt.title(model_name)
    plt.show()
    return macro_f1, accuracy