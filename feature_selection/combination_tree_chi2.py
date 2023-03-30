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

from datetime import datetime

import csv

tree_short = set(['402.8', '411', '764.1', '772.3', '737.4', '706.6', '753.8', '813.4', '690.2'])
tree_lists = set(['402.8', '411', '764.1', '749.7', '774.4', '778.5', '745.6', '565', '552.7', '811.3', '801.1', '671.7', '870.9', '626.6'])
ch2_lists = set(['762.1', '747.7', '753.8', '788.7', '655.3', '879.1', '774.4', '540.3', '823.6', '565', '597.8', '394.6'])
comb_lists = set(['747.7', '762.1', '753.8', '788.7', '655.3', '879.1', '774.4', '540.3', '823.6', '565', '597.8', '394.6'])

rfe_randomforest_lists = set(['762.1', '737.4', '721', '770.3', '753.8', '396.7'])
rfe_svc_lists = set(['747.7', '762.1', '753.8', '788.7', '655.3', '879.1', '774.4', '540.3', '823.6', '565', '597.8', '394.6'])
corr_pr_lists = set(['411', '404.9', '762.1', '749.7', '774.4', '778.5', '731.3', '696.4', '817.5', '885.2', '803.1', '675.8'])
sfs_forward_lists = set(['404.9', '411', '762.1', '749.7', '774.4', '778.5', '731.3', '696.4', '817.5', '885.2', '803.1', '675.8'])
ridge_lists = set(['562.9', '762.1', '747.7', '768.2', '776.4', '538.3', '827.7', '872.9', '690.2', '575.2'])
lasso_lists = set(['772.3', '792.8', '757.9', '766.2', '875', '823.6'])

full_lists = set.union(tree_short,tree_lists,ch2_lists,comb_lists, 
                rfe_randomforest_lists, rfe_svc_lists, corr_pr_lists, 
                sfs_forward_lists, ridge_lists, lasso_lists)

full_lists = list(full_lists) 

def process_data(file_name, sheet_name, selected_bands):
    full_data = pd.read_excel(file_name, sheet_name= sheet_name)

    str_cols = full_data.columns.astype(str)
    full_data.columns = str_cols
    cols = str_cols.tolist()
    cols = cols[1:]

    _input = full_data[cols]
    label = full_data['label'].tolist()
    label = np.asarray(label).reshape(-1,)

    print()
    print("Start to find best six bands via combination method ................")

    t2 = time()

    clf = QuadraticDiscriminantAnalysis()
    cv = RepeatedKFold(n_splits=4,n_repeats=4, random_state=46)

    _best_score = 0.0
    _best_band = set()

    #comb = combinations(_index_list, 6)
    with open("comb_6_test.csv", newline='') as f:
        f_reader = csv.reader(f, delimiter=',')
        count = 0
        for row in f_reader:
            _bands = [selected_bands[int(i)] for i in row]
            _new_input = _input[_bands]
            scores = cross_val_score(clf, _new_input, label, scoring='accuracy', cv=cv, n_jobs=-1)
            _score = np.mean(scores)
            if _score > _best_score:
                _best_score = _score
                _best_band = _bands

            count += 1
            if count % 5000 == 0:
                print("Now is {} and using {} seconds".format(count, time()-t2))
                t2 = time()

    print("Using {} sec to do combination".format(time()-t2))
    print(_best_band)
    print(_best_score)


    six_bands_file = "tree_chi_comb_6_bands.pkl"
    with open(six_bands_file,'wb') as output_f:
        pickle.dump(_best_band, output_f)

if __name__ == "__main__":
    file_name = 'Adjusted categorization - 1-5,16-30, 35-70.xlsx'
    sheet_name='H vs. 1-15'

    print("Start time is {}".format(datetime.now()))
    print()
    process_data(file_name, sheet_name, full_lists)
    print()
    print("End time is {}".format(datetime.now()))
