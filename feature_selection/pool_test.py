import numpy as np
import pickle
import pandas as pd
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RepeatedKFold
from datetime import datetime
import csv

from multiprocessing import Pool
from functools import partial


def process_data(selected_bands,_input, label, csv_file):
    print()
    print("Start to find best six bands for {} .....".format(csv_file))

    t2 = time()

    clf = QuadraticDiscriminantAnalysis()
    #cv = RepeatedKFold(n_splits=5,n_repeats=2, random_state=46)

    _best_score = 0.0
    _best_band = set()

    #comb = combinations(_index_list, 6)
    with open(csv_file, newline='') as f:
        f_reader = csv.reader(f, delimiter=',')
        count = 0
        for row in f_reader:
            print('Get row {}'.format(row))
            _bands = [selected_bands[int(i)] for i in row]
            print('Get _bands {}'.format(_bands))
            _new_input = _input[_bands]
            scores = cross_val_score(clf, _new_input, label, scoring='accuracy', cv=4)
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


    six_bands_file = csv_file[:-4] +  "_bands.pkl"
    with open(six_bands_file,'wb') as output_f:
        pickle.dump(_best_band, output_f)

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

comb_6_list = ['comb_6_1.csv', 'comb_6_2.csv', 'comb_6_3.csv', 'comb_6_4.csv', 'comb_6_5.csv', 'comb_6_6.csv', 'comb_6_7.csv', 'comb_6_8.csv',
            'comb_6_9.csv', 'comb_6_10.csv', 'comb_6_11.csv', 'comb_6_12.csv', 'comb_6_13.csv', 'comb_6_14.csv', 'comb_6_15.csv', 'comb_6_16.csv',]

if __name__ == "__main__":
    file_name = 'Adjusted categorization - 1-5,16-30, 35-70.xlsx'
    sheet_name='H vs. 1-15'
    full_data = pd.read_excel(file_name, sheet_name= sheet_name)

    str_cols = full_data.columns.astype(str)
    full_data.columns = str_cols
    cols = str_cols.tolist()
    cols = cols[1:]

    _input = full_data[cols]
    label = full_data['label'].tolist()
    label = np.asarray(label).reshape(-1,)
    print("Start time is {}".format(datetime.now()))
    print()

    with Pool(10) as p:
        temp = partial(process_data, full_lists, _input, label)
        p.map(func=temp, iterable=comb_6_list)

    print()
    print("End time is {}".format(datetime.now()))