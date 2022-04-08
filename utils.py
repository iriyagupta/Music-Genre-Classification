import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
plt.rcParams["figure.figsize"] = (20,8)


def split_scale_data_for_tunned_models(X, y):
    ## Data Spiltting
    X_dev_t, X_test_t, y_dev_t, y_test_t = train_test_split(X, y, test_size = 0.1, stratify = y, shuffle = True, random_state = 42) 
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_dev_t, y_dev_t, test_size=0.1, stratify = y_dev_t, shuffle = True, random_state= 42) 

    ##Data Standardization using MinMaxScaler
    minmax_t = MinMaxScaler()
    X_train_t = minmax_t.fit_transform(X_train_t)
    X_val_t = minmax_t.transform(X_val_t)
    X_test_t = minmax_t.transform(X_test_t)
    
    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t

def split_scale_data_for_untunned_models(X, y):
    # Data Spiltting
    X_train_ut, X_test_ut, y_train_ut, y_test_ut = train_test_split(X, y, test_size=0.1, stratify=y, shuffle = True, random_state= 42) 

    # Data Standardization using MinMaxScaler
    minmax_ut = MinMaxScaler()
    X_train_ut = minmax_ut.fit_transform(X_train_ut)
    X_test_ut = minmax_ut.transform(X_test_ut)
    return X_train_ut, X_test_ut, y_train_ut, y_test_ut


def evaluate_models(y_pred, y_true, class_labels, model_name = ''):
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

