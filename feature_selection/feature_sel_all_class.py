import numpy as np
import pickle
import pandas as pd
from time import time
from datetime import datetime
from itertools import combinations

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest,f_classif,chi2

from sklearn.linear_model import RidgeClassifier,Lasso

#from sklearn.ensemble import StackingClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import RFE
#import xgboost as xgb

def get_top_bands_comb(best_band,remain_bands, _input, label, model):

    print("Start to process bands selection .....................")
    print("#######################################################")
    t0 = time()
    results = list()
    _best_score = 0.0
    cv = RepeatedKFold(n_splits=4,n_repeats=4, random_state=46)

    for i in range(len(remain_bands)):
        #t1 = time()
        _best_band = ' '
        result = dict()
        _count = 0
        for item in remain_bands:
            _count += 1
            selected_bands = best_band.copy()
            selected_bands.append(item)
            _new_input = _input[selected_bands]
            scores = cross_val_score(model, _new_input, label, scoring='accuracy', cv=cv, n_jobs=-1)
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
        #print("Round {} process {} bands, get score {} using {} seconds ".format(i, _count, _best_score, time()-t1))

    print("Using {} sec".format(time()-t0))
    return results

if __name__ == "__main__":
    _file_name = 'Adjusted categorization - 1-5,16-30, 35-70.xlsx'
    full_data = pd.read_excel(_file_name, sheet_name='Multiclasses')

    print("Start time is {}".format(datetime.now()))
    start = datetime.now()

    print("Load data success")

    str_cols = full_data.columns.astype(str)
    full_data.columns = str_cols
    cols = str_cols.tolist()
    cols = cols[1:]

    __input = full_data[cols]
    label = full_data['label'].tolist()
    label = np.asarray(label).reshape(-1,)

    scaler = MinMaxScaler()
    _std = scaler.fit_transform(__input)
    _input = pd.DataFrame(_std,columns=cols)

    # chi2
    clf = SelectKBest(f_classif, k=2)
    X_new = clf.fit_transform(_input, label)
    indices = np.argsort(clf.scores_)[::-1]
    top_bands = list(_input.columns.values[indices])
    print("Top bands: {}".format(top_bands[:10]))

    # Lasso
    # lasso = Lasso(alpha=1).fit(_input, label)
    # indices = np.argsort(lasso.coef_)[::-1]
    # top_bands = list(_input.columns.values[indices])
    # print(top_bands[:10])

    # RidgeClassifier
    # rcf = RidgeClassifier(alpha=1).fit(_input, label)
    # indices = np.argsort(rcf.coef_)[::-1]
    # top_bands = list(_input.columns.values[indices][0])
    # print(top_bands[:10])

    # R_regression
    # _corr = r_regression(_input, label)
    # indices = np.argsort(_corr)[::-1]
    # top_bands = list(_input.columns.values[indices])
    # print(top_bands[:10])

    # RFE with Random Forest
    #clf = RandomForestClassifier(random_state=46)
    #clf = SVC(kernel="linear")
    # rfe = RFE(estimator=clf, step=1, n_features_to_select=3)
    # rfe.fit(_input, label)

    # _mask = np.ma.masked_where(rfe.ranking_ == 1, rfe.ranking_)
    # top_bands = list(_input.columns.values[_mask.mask])
    # print(top_bands[:10])


    # Tree method
    # print("Start to get bands via Tree method")
    # etc = ExtraTreesClassifier(n_estimators=200, n_jobs=-1)
    # etc = etc.fit(_input, label)
    # #indices = np.argsort(etc.feature_importances_)[::-1]
    # #sfm = SelectFromModel(etc)
    # #X_new = sfm.fit_transform(_input, label)

    # # Model selected 93 bands
    # col_importance = []

    # for col, importance in zip(cols,etc.feature_importances_):
    #     item= dict()
    #     item['col'] = col
    #     item['importance'] = importance
    #     col_importance.append(item)

    # pd_col_imp = pd.DataFrame(col_importance)
    # print("Start to cal best bands")
    # #_number = X_new.shape[1]
    # _number = 80
    # top_bands = pd_col_imp.sort_values(by=['importance'], ascending=False).head(_number)['col'].tolist()
    # print(top_bands[:10])


    best_band = [top_bands[0]]
    remain_bands = top_bands[1:]
    # remain_bands = cols.copy()
    # remain_bands.remove(best_band[0])

    model =  QuadraticDiscriminantAnalysis()
    results = get_top_bands_comb(best_band, remain_bands, _input, label, model)
    print(results[-1]['score'])
    print(results[-1]['bands'])

    _result = pd.DataFrame(results)

    _result.to_csv('Multi_classes_SVC_linear_bands.csv', float_format='%.5f',index=False)

    print("Start time is {}, End time is {}".format(start,datetime.now()))