__author__ = 'dario'



from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
import random
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.preprocessing import Imputer

import pandas as pd


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

def build_classifiers():
    ESTIMATORS = {
        "sgd": SGDClassifier(loss="modified_huber", penalty="elasticnet", n_iter=20000, class_weight='auto', alpha=0.1, epsilon=0.01),
        "Extra trees": ExtraTreesClassifier(n_estimators=3000, max_depth=None, min_samples_leaf=1, random_state=0),
        "Logistic regression": LogisticRegression(),
        "Random forest": RandomForestClassifier(n_estimators=400, n_jobs=-1),
        }

    return ESTIMATORS

def build_xgboost_regressor(X_test, X_train_full, y_train_full):

    #print "Building xgboost regressor..."

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
    offset = int(len(X_train_full) * 0.2)
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

    #print "Building random forest regressor..."

    rf = RandomForestRegressor(n_estimators=800)
    probas_rf = rf.fit(X_train_full, y_train_full).predict(X_test)
    return probas_rf

def build_sgd_regressor(X_test, X_train_full, y_train_full):

    #print "Building SGD regressor..."

    rf = SGDRegressor(loss="modified_huber", penalty="elasticnet", n_iter=20000, alpha=0.1, epsilon=0.01)
    probas_rf = rf.fit(X_train_full, y_train_full).predict(X_test)
    return probas_rf


def build_regressors():
    ESTIMATORS = {
        #"sgd": build_sgd_regressor,
        "xgboost": build_xgboost_regressor,
        "Random forest":  build_random_forest_regressor,
        }

    return ESTIMATORS


''''
trainFilename = '/home/dario/Dropbox/Datasets/KDDcup98/cup98LRN.txt'
file_io = open(trainFilename, 'r')
csvData = csv.reader(file_io, delimiter=',')
header = csvData.next()
file_io.close()

print("\nHeader = {}\n".format(header))


#create the training & test sets, skipping the header row with [1:]
dataSet = genfromtxt(open(trainFilename, 'r'), delimiter=',', dtype=None)[1:]
print("\nData dimension = {}\n".format(dataSet.shape))

for i in range(0,10):
    print dataSet[i]


'''


# DATA PREPARATION VARIABLES


sampling_percentage = 0.1
no_features_after_feat_selection_1 = 40
no_features_after_feat_selection_2 = 30


#using error_bad_lines=False will cause the offending lines to be skipped
#df = pd.read_csv('/home/dario/Dropbox/Datasets/KDDcup98/cup98LRN.txt', usecols=columns_to_use, sep=",", error_bad_lines=False)
df = pd.read_csv('/home/dario/Dropbox/Datasets/KDDcup98/cup98LRN.txt', sep=",", error_bad_lines=False)

target_name_binary = "TARGET_B"
target_name_continuous = "TARGET_D"

target_b_full = df[target_name_binary]
target_d_full = df[target_name_continuous]
df.drop([target_name_binary], axis=1, inplace=True) #drop the binary target
df.drop([target_name_continuous], axis=1, inplace=True)


column_names = df.columns

print "sampling..."
# Sample the data. It is so big we don't need all of it.
size = int(len(df) * sampling_percentage)
rows = random.sample(df.index, size)
train = pd.DataFrame(df.ix[rows], columns=column_names)
target_b = target_b_full.ix[rows]
target_d = target_d_full.ix[rows]
hold_out = pd.DataFrame(df.drop(rows), columns=column_names)
hold_out_target_b = target_b_full.drop(rows)
hold_out_target_d = target_d_full.drop(rows)


print "calculating the correlation..."
# Calculating the correlation
train[target_name_continuous] = target_d #appending back the target class, as the correlation below happen within the matrix
corr = train.corr()[target_name_continuous][train.corr()[target_name_continuous] < 1].abs()
train.drop([target_name_continuous], axis=1, inplace=True) # deleting again the target column, I know it is ugly...
corr.sort(ascending=True)
print "WORSE 5 CORRELATED FEATURES:"
print corr.head()

corr.sort(ascending=False)
print "TOP 5 CORRELATED FEATURES:"
print corr.head()


# Select the top correlated features for the next steps
top_correlated_features = corr.index[[range(0, no_features_after_feat_selection_1)]].values
train = train[top_correlated_features]
hold_out = hold_out[top_correlated_features]

print "Data-set shape: {}".format(df.shape)

print "replacing strings with numbers..."
# Replace text data with numeric data
# Manually checked that below simplifications work
dtypes = train.columns.to_series().groupby(train.dtypes).groups
dtypes2 = {k.name: v for k, v in dtypes.items()}
if dtypes2.__contains__('object'):
    for column in dtypes2['object']:
        dic = {}
        i=1
        for val in train[column].unique():
            dic[val] = i
            i+=1
        train[column] = train[column].map(lambda x: dic[x])
        hold_out[column] = hold_out[column].map(lambda x: dic[x])
else :
    print "not performed!"




print "replacing NaNs..."
#Replace the NaNs with the mean value in column
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
imp.fit(hold_out)
cleaned_data = pd.DataFrame(imp.fit_transform(train), columns=train.columns)
hold_out = pd.DataFrame(imp.transform(hold_out), columns=train.columns)


print "Data-set shape: {}".format(cleaned_data.shape)

'''
# Feature selection using random forest
print "building random forest..."
rf = RandomForestRegressor(n_estimators=400, verbose=True, n_jobs=-1)
rf.fit(cleaned_data, target_d)

importance = rf.feature_importances_
feature_ranking = pd.Series(importance, index=cleaned_data.columns)
feature_ranking.sort(ascending=False)
print "Feature importance:"

print feature_ranking

# Reduce the feature to 30
final_selected_features = feature_ranking.index[[range(0, no_features_after_feat_selection_2)]].values
'''
final_selected_features = cleaned_data.columns

final_train_data =  np.array(cleaned_data[final_selected_features])
target_b = np.array(target_b)

print "Data-set shape: {}".format(final_train_data.shape)

'''
# Binary classification
estimators = build_classifiers()
estimators_names = estimators.keys()
n_folds = 5
cv = cross_validation.StratifiedKFold(target_b, n_folds=n_folds)
results = pd.DataFrame(index=range(0,n_folds), columns=estimators_names)
index = 0
for train_indices, test_indices in cv:
    X_train, X_test = final_train_data[train_indices], final_train_data[test_indices]
    y_train, y_test = target_b[train_indices], target_b[test_indices]

    for name, estimator in estimators.items():
        #print "I am trying to build a {}".format(name)
        probas = estimator.fit(X_train, y_train).predict_proba(X_test)
        auc_score = roc_auc_score(y_test, [x[1] for x in probas])
        print "AUC score for {} on fold {} is: {}".format(name, index, auc_score)
        results[name][index] = auc_score

    #print "AUC score for fold {}: {}".format(index, results_on_fold)
    index += 1

# now print out the mean of the cross-validated results
print "All results: {}".format(results)
results_mean = results.mean()
for estimator in estimators_names:
    print "Average AUC for {} is: {}".format(estimator, results_mean[estimator])

# comparing the performances from CV with the hold out data
for name, estimator in estimators.items():
    #print "I am trying to build a {}".format(name)
    hold_out_probas = estimator.fit(final_train_data, target_b).predict_proba(hold_out[final_selected_features])
    hold_out_auc_score = roc_auc_score(hold_out_target_b, [x[1] for x in hold_out_probas])
    print "AUC score for {} on hold out data is: {}".format(name, index, hold_out_auc_score)

'''





# Continuous target
target_d = np.array(target_d)
estimators = build_regressors()
estimators_names = estimators.keys()
n_folds = 5
cv = cross_validation.KFold(len(target_d), n_folds=n_folds)
results = pd.DataFrame(index=range(0,n_folds), columns=estimators_names)
index = 0
for train_indices, test_indices in cv:
    X_train, X_test = final_train_data[train_indices], final_train_data[test_indices]
    y_train, y_test = target_d[train_indices], target_d[test_indices]

    for name, estimator in estimators.items():
        #print "I am trying to build a {}".format(name)
        y_out = estimator(X_test, X_train, y_train)
        estimator_gini = normalized_gini(y_test, y_out)
        print "GINI score for {} on fold {} is: {}".format(name, index, estimator_gini)
        results[name][index] = gini

    index += 1

# now print out the mean of the cross-validated results
print "All results: {}".format(results)
results_mean = results.mean()
for estimator in estimators_names:
    print "Average GINI for {} is: {}".format(estimator, results_mean[estimator])

# comparing the performances from CV with the hold out data
for name, estimator in estimators.items():
    #print "I am trying to build a {}".format(name)
    y_out = estimator(hold_out[final_selected_features], final_train_data, target_d)
    estimator_gini = normalized_gini(hold_out_target_d, y_out)

    print "GINI score for {} on hold out data is: {}".format(name, index, estimator_gini)
