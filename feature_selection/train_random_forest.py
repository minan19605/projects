
import numpy as np
import os
from time import time
import pandas as pd
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import make_moons, make_circles, make_classification
#from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process.kernels import RBF
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

file_list = ['More H VS. SC 50-70.xlsx']
t0 = time()
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
        t1 = time()
        print("Start to process band: {}".format(col))
        x = new_pd[col].tolist()
        x = np.asarray(x).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.3, random_state=66)
        
        param_grid = {'n_estimators':[50,100,150],'max_depth': range(1,20), 'criterion':['gini','entropy']}
        #print("Set Grid parameters")
        clf = GridSearchCV(
            RandomForestClassifier(), param_grid, cv=10)
        #print("Start to fit")
        clf = clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        
        result_score = dict()
        result_score['WaveLength'] = col
        result_score['score']= score
        result_score['best_parms'] = clf.best_estimator_
        result_scores.append(result_score)
        print("Process band {} spend {} seconds".format(col, time()-t1))
        
    new_file = 'sc50_70_new_random_forests_result.xlsx'
    result_pd = pd.DataFrame(result_scores)
    result_pd.to_excel(new_file)

    print("Total spend {} seconds".format(time()-t0))
