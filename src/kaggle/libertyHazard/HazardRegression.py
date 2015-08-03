from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

__author__ = 'dario'

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

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

#Read the data in
df = pd.read_csv("/home/dario/LogicalGlueDatasets/Liberty_Mutual_group_hazard_prediction/train.csv", sep=",", index_col=0)
test = pd.read_csv("/home/dario/LogicalGlueDatasets/Liberty_Mutual_group_hazard_prediction/test.csv", sep=",", index_col=0)

ids = test.index

print "Data frame shape: {}".format(df.shape)

full_target = df.Hazard
df.drop('Hazard', axis=1, inplace=True)

df.drop('T1_V13', axis=1, inplace=True)
df.drop('T1_V10', axis=1, inplace=True)

train_cols = df.columns

# get the categorical columns
fact_cols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11',
             'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12',
             'T2_V13']

# Finding the numerical columns
numerical_cols = []
for col in train_cols:
    if not fact_cols.__contains__(col):
        numerical_cols.append(col)

# Selecting the train data set with numerical columns only
full_data_set = df[numerical_cols]
# Appending the pivoted categorical column to the train data set
for categorical_column in fact_cols:
    dummy_ranks = pd.get_dummies(df[categorical_column], prefix=categorical_column)
    full_data_set = full_data_set.join(dummy_ranks)

print "Training data shape after dummies: {}".format(full_data_set.shape)



# The regressor
rf = RandomForestRegressor(n_estimators=200)
#gbr = GradientBoostingRegressor()
#lr = LogisticRegression()

# normalized_gini(desired_y, predicted_y)

print "Cross validation is running..."
cv = cross_validation.KFold(len(full_target), n_folds=5, shuffle=True)

results = []
train_array = np.asanyarray(full_data_set)
target_array = np.asanyarray(full_target)
index = 1
for traincv, testcv in cv:
    #print "waiting for Logistic regression..."
    #probas_lr = lr.fit(train_array[traincv], target_array[traincv]).predict(train_array[testcv])
    print "waiting for random forest regression..."
    probas_rf = rf.fit(train_array[traincv], target_array[traincv]).predict(train_array[testcv])
    #print "waiting for gradient boosting regression..."
   # probas_gbr = gbr.fit(train_array[traincv], target_array[traincv]).predict(train_array[testcv])

    #probas = np.average(probas_lr, probas_rf, probas_gbr)
    probas = probas_rf

    gini_score = normalized_gini(target_array[testcv], probas)
    print "GINI score for fold {}: {}".format(index, gini_score)
    index += 1
    results.append(gini_score)

# now print out the mean of the cross-validated results
print "Avg gini across all folds: " + str(np.array(results).mean())

'''
# NOW ON TEST SET
# Train the regressor on the full data set
reg.fit(full_data_set, full_target)
print "\n Creating the submission file..."


print "\nTest data shape: {}".format(np.shape(test))

# Appending the pivoted categorical column to the train data set
for categorical_column_test in fact_cols:
    dummy_ranks_test = pd.get_dummies(test[categorical_column_test], prefix=categorical_column_test)
    test = test.join(dummy_ranks_test)

print "\nTest data shape after dummies: {}".format(np.shape(test))

# Add extra features to test set
add_extra_features(test, list(numerical_cols))
print "\nTest data shape after extra features: {}".format(np.shape(test))

pred = reg.predict(test)

preds = pd.DataFrame({"Id": ids, "Hazard": pred})
preds = preds[['Id', 'Hazard']]
preds.to_csv('submit.csv', index=False)

'''



print "\nFinish!"

