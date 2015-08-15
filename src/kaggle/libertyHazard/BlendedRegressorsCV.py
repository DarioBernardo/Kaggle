
__author__ = 'dario'

'''
This benchmark uses xgboost and early stopping to achieve a score of 0.38019
In the liberty mutual group: property inspection challenge

Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
import xgboost as xgb
from sklearn import cross_validation

def gini(solution, submission):
    df = sorted(zip(solution, submission),
                key=lambda x: x[1], reverse=True)
    random = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = np.sum([x[0] for x in df])
    cumPosFound = np.cumsum([x[0] for x in df])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [l - r for l, r in zip(Lorentz, random)]
    return np.sum(Gini)


def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini


def build_xgboost_regressor(X_test, X_train_full, y_train_full):

    print "Building xgboost regressor..."

    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["min_child_weight"] = 5
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.8
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 8
    plst = list(params.items())
    num_rounds = 10000
    # Using 4000 rows for early stopping.
    offset = 4000
    # Construct Training matrix and early stopping matrix
    xgtrain = xgb.DMatrix(X_train_full[offset:, :], label=y_train_full[offset:])
    xgval = xgb.DMatrix(X_train_full[:offset, :], label=y_train_full[:offset])
    # Construct matrix for test set
    xgtest = xgb.DMatrix(X_test)
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    y_out = model.predict(xgtest)
    return y_out

def build_random_forest_regressor(X_test, X_train_full, y_train_full):

    print "Building random forest regressor..."

    rf = RandomForestRegressor(n_estimators=800)
    probas_rf = rf.fit(X_train_full, y_train_full).predict(X_test)
    return probas_rf

def build_logistic_regressor(X_test, X_train_full, y_train_full):

    print "Building logistic regressor..."
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_train_full, y_train_full)
    logreg_predict = logreg.predict(X_test)

    return logreg_predict

def build_extra_tree_regressor(X_test, X_train_full, y_train_full):


    print "Building ExtraTrees regressor..."
    etr = ExtraTreesRegressor(n_estimators=500)
    etr.fit(X_train_full, y_train_full)
    etr_predict = etr.predict(X_test)

    return etr_predict

def build_estimators():
    ESTIMATORS = {
        "xgb": build_xgboost_regressor,
        "Extra trees": build_extra_tree_regressor,
        #"Logistic regression": build_logistic_regressor,
        "Random forest": build_random_forest_regressor,
        }

    return ESTIMATORS

def run_cross_validation(train,labels,test,estimators):

    cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True)
    results = []
    index = 1
    for train_indices, test_indices in cv:
        X_train_full, X_test = train[train_indices], train[test_indices]
        y_train_full, y_test = labels[train_indices], labels[test_indices]

        y_out = []
        for name, estimator in estimators.items():
            #print "I am trying to build a {}".format(name)
            y_out_temp = estimator(X_test, X_train_full, y_train_full)
            gini_temp = normalized_gini(y_test, y_out_temp)
            print "{} score is: {}\n".format(name, gini_temp)
            #print "{} predictions are: {}\n".format(name, y_out_temp)
            if len(y_out) == 0:
                y_out = y_out_temp
            else:
                y_out += y_out_temp

        gini = normalized_gini(y_test, y_out)
        print "All AUC scores: {}".format(gini)

        print "AUC score for fold {}: {}".format(index, gini)
        index += 1
        results.append(gini)

    # now print out the mean of the cross-validated results
    print "All results: {}".format(results)
    print "Avg auc across all folds: " + str(np.array(results).mean())



#load train and test
train = pd.read_csv('~/Datasets/Liberty_Mutual_group_hazard_prediction/train.csv', index_col=0)
test = pd.read_csv('~/Datasets/Liberty_Mutual_group_hazard_prediction/test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)


columns = train.columns
test_ind = test.index


train_s = train
test_s = test


# Drop some bad features
train_s.drop('T2_V10', axis=1, inplace=True)
train_s.drop('T2_V7', axis=1, inplace=True)
train_s.drop('T1_V13', axis=1, inplace=True)
train_s.drop('T1_V10', axis=1, inplace=True)

test_s.drop('T2_V10', axis=1, inplace=True)
test_s.drop('T2_V7', axis=1, inplace=True)
test_s.drop('T1_V13', axis=1, inplace=True)
test_s.drop('T1_V10', axis=1, inplace=True)


train_s = np.array(train_s)
test_s = np.array(test_s)
labels = np.array(labels)

# label encode the categorical variables
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:,i]) + list(test_s[:,i]))
    train_s[:,i] = lbl.transform(train_s[:,i])
    test_s[:,i] = lbl.transform(test_s[:,i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)

run_cross_validation(train_s,labels,test_s, build_estimators())

#preds1 = xgboost_pred(train_s,labels,test_s)

'''
#model_2 building
print "Prediction 2..."

train = train.reset_index().T.to_dict().values()
test = test.reset_index().T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

preds2 = xgboost_pred(train,labels,test)


preds = 0.6 * preds1 + 0.4 * preds2






#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('xgboost_benchmark_kk3.csv')
'''



print "Done!"

