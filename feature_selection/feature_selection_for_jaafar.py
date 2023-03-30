import numpy as np
import pickle
import pandas as pd
from time import time
from itertools import combinations

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

#from sklearn.datasets import make_moons, make_circles, make_classification
#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import StackingClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def build_model():
    estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svc', SVC(kernel = 'sigmoid',gamma='auto')),
    ('knc',KNeighborsClassifier(n_neighbors=3)),
    ('GNB',GaussianNB()),
    ('QDA',QuadraticDiscriminantAnalysis()),
    ('DTC',DecisionTreeClassifier())
    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator= SGDClassifier(), cv=10, #LogisticRegression
        n_jobs = 18
    )
    
    return clf

def process_data(file_name, sheet_name):
    full_data = pd.read_excel(file_name, sheet_name= sheet_name)

    str_cols = full_data.columns.astype(str)
    full_data.columns = str_cols
    cols = str_cols.tolist()
    cols = cols[1:]

    _input = full_data[cols]
    label = full_data['label'].tolist()
    label = np.asarray(label).reshape(-1,)

    etc = ExtraTreesClassifier(n_estimators=200)
    etc = etc.fit(_input, label)

    sfm = SelectFromModel(etc)
    X_new = sfm.fit_transform(_input, label)
    selected_bands_num = X_new.shape[1]

    col_importance = []

    for col, importance in zip(cols,etc.feature_importances_):
        item= dict()
        item['col'] = col
        item['importance'] = importance
        col_importance.append(item)

    pd_col_imp = pd.DataFrame(col_importance)
    top_bands = pd_col_imp.sort_values(by=['importance'], ascending=False).head(selected_bands_num)['col'].tolist()

    clf = QuadraticDiscriminantAnalysis()
    cv = RepeatedKFold(n_splits=4,n_repeats=4, random_state=46)

    best_band = [top_bands[0]]
    remain_bands = top_bands[1:]

    print("Start to process bands selection.....................")
    print("#######################################################")
    t0 = time()
    results = list()
    _best_score = 0.0

    for i in range(len(remain_bands)):
        _best_band = ' '
        result = dict()
        for item in remain_bands:
            selected_bands = best_band.copy()
            selected_bands.append(item)
            _new_input = _input[selected_bands]
            scores = cross_val_score(clf, _new_input, label, scoring='accuracy', cv=cv, n_jobs=-1)
            _score = np.mean(scores)

            if _score > _best_score:
                _best_score = _score
                _best_band = item

        if _best_band == ' ':  # Can't find more bands to increase score, then exit 
            break

        best_band.append(_best_band)
        remain_bands.remove(_best_band)
        result['score'] = _best_score
        result['bands'] = best_band.copy()

        results.append(result)
        
    print("Using {} sec to get best bands".format(time()-t0))
    print(best_band)
    print(_best_score)

    #bands_pd = pd.DataFrame(results)

    clf = build_model()
    selected_bands = results[-1]['bands']

    X = _input[selected_bands].to_numpy()
    y = label

    print("Start to cal best_bands f1 score .....................")
    print("#######################################################")
    t1 = time()
    rkf = RepeatedKFold(n_splits=4, n_repeats=4, random_state=2652124)
    f1_scores = []
    for train_index, test_index in rkf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        _score = f1_score(y_test, y_pred, average=None)
        f1_scores.append(_score)

    f1_scores = np.asarray(f1_scores)

    print("Using {} sec to get best scores".format(time()-t1))

    f1_scores_file = sheet_name +'_selected_bands_f1_scores.pkl'

    best_bands_file = sheet_name+ '_selected_bands.pkl'

    with open(best_bands_file,'wb') as output_f:
        pickle.dump(selected_bands, output_f)

    with open(f1_scores_file, "wb") as output_f:
        pickle.dump(f1_scores,output_f)

    print()
    print("Write selected bands and f1-scores")

    if len(selected_bands) > 6:
        print()
        print("Start to find best six bands via combination method ................")

        t2 = time()

        clf = QuadraticDiscriminantAnalysis()
        cv = RepeatedKFold(n_splits=4,n_repeats=4, random_state=46)

        _best_score = 0.0
        _best_band = set()

        comb = combinations(selected_bands, 6)
        for item in comb:
            _bands = set(item) # item is tuple, convert to set
            result = dict()
            #print(bands_set)
            _new_input = _input[_bands]
            scores = cross_val_score(clf, _new_input, label, scoring='accuracy', cv=cv, n_jobs=-1)
            _score = np.mean(scores)
            if _score > _best_score:
                _best_score = _score
                _best_band = _bands

        print("Using {} sec to do combination".format(time()-t2))
        print(_best_band)
        print(_best_score)


        clf = build_model()

        X = _input[_best_band].to_numpy()
        y = label

        print("Start to cal best six bands f1 score .....................")
        print("#######################################################")
        t3 = time()
        rkf = RepeatedKFold(n_splits=4, n_repeats=4, random_state=2652124)
        f1_scores_6 = []
        for train_index, test_index in rkf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            _score = f1_score(y_test, y_pred, average=None)
            f1_scores_6.append(_score)

        f1_scores_6 = np.asarray(f1_scores_6)

        f1_scores_file_6 = sheet_name +'_6_bands_f1_scores.pkl'

        six_bands_file = sheet_name+ '_6_bands.pkl'

        with open(six_bands_file,'wb') as output_f:
            pickle.dump(_best_band, output_f)

        with open(f1_scores_file_6, "wb") as output_f:
            pickle.dump(f1_scores_6,output_f)

        print("Using {} sec to get six bands scores and write to files".format(time()-t3))

if __name__ == "__main__":
    file_name =  'Adjusted categorization - 1-5,16-30, 35-70.xlsx' # 'An-3 classis.xlsx'
    #sheet_list = ['All classes', 'H vs. Sc 16-49','H vs. 50-70','H vs. Sc 1-15']
    xl = pd.ExcelFile(file_name)
    sheet_list = xl.sheet_names
    sheet_list.remove('All categories')
    sheet_list.remove('VEG index')

    for sheet_name in sheet_list:
        print("Process sheet {}".format(sheet_name))
        print()
        process_data(file_name, sheet_name)
        print()