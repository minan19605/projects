
import numpy as np
import os
from time import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

def build_model(x_train, y_train, x_test, y_test):

    param_grid = {'C': [1e2,5e2,1e3, 5e3, 1e4],
              'gamma': [0.001, 0.005, 0.01,0.05,0.1,0.2]}
    clf = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(x_train, y_train)

    score = clf.score(x_test, y_test)
    return score, clf.best_estimator_

#file_list = ['sc1_new.xlsx', 'sc2_4_new.xlsx', 'sc5_7_new.xlsx', 'sc8_15_new.xlsx','sc16_29_new.xlsx','sc30_49_new.xlsx','sc50_70_new.xlsx']
file_list = ['More H VS. SC 50-70.xlsx']

for file in file_list:
    full_data = pd.read_excel(file)
    health_rows = full_data[full_data['label'] == 0]
    disease_rows = full_data[full_data['label'] != 0]
    length = disease_rows['label'].count()
    selected_rows = health_rows.sample(length)
    new_pd = selected_rows.append(disease_rows, ignore_index=True)
    label = new_pd['label'].tolist()
    label = np.asarray(label).reshape(-1,)
    
    cols = new_pd.columns.tolist()
    cols = cols[1:]
    
    # Process each wavelenth
    result_scores = []
    for col in cols:
        x = new_pd[col].tolist()
        x = np.asarray(x).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.3, random_state=66)
        
        #t0 = time()
        param_grid = {'kernel':['linear','poly','rbf','sigmoid'],'C': [1e2,5e2,1e3, 5e3, 1e4],
                      'gamma': [0.001, 0.005, 0.01,0.05,0.1,0.2]}
        #print("Set Grid parameters")
        clf = GridSearchCV(
            SVC(class_weight='balanced'), param_grid, cv=10)
        #print("Start to fit")
        clf = clf.fit(x_train, y_train)
        #print(clf.best_estimator_)
        
        #clf = AdaBoostClassifier(n_estimators=100, random_state=46)
        #clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        
        result_score = dict()
        result_score['WaveLength'] = col
        result_score['score']= score
        result_score['best_parms'] = clf.best_estimator_
        result_scores.append(result_score)
        
    new_file = file.split('.')[0] + '_svm_result.xlsx'
    result_pd = pd.DataFrame(result_scores)
    result_pd.to_excel(new_file)
