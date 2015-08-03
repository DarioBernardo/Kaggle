__author__ = 'dario'

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer as DV

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

train = pd.read_csv('/home/dario/LogicalGlueDatasets/Liberty_Mutual_group_hazard_prediction/train.csv')
test = pd.read_csv('/home/dario/LogicalGlueDatasets/Liberty_Mutual_group_hazard_prediction/test.csv')

ids = test['Id']
y = train['Hazard']
train = train.drop(['Hazard', 'Id'], axis=1)
test = test.drop(['Id'], axis=1)
train_columns = train.columns

# get the categorical columns
fact_cols = ['T1_V4', 'T1_V5', 'T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11',
             'T1_V12', 'T1_V15', 'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12',
             'T2_V13']
# Finding the numerical columns
numerical_cols = []
for col in train_columns:
    if not fact_cols.__contains__(col):
        numerical_cols.append(col)

fact_train = train[fact_cols]
fact_test = test[fact_cols]

column_names_before_non_linearity = train.columns


#put the numerical as matrix
num_train_data = train.drop(fact_cols, axis=1).as_matrix()
num_test_data = test.drop(fact_cols, axis=1).as_matrix()

#transform the categorical to dict
dict_train_data = fact_train.T.to_dict().values()
dict_test_data = fact_test.T.to_dict().values()

#vectorize
vectorizer = DV(sparse = False)
vec_train_data = vectorizer.fit_transform(dict_train_data)
vec_test_data = vectorizer.fit_transform(dict_test_data)

#merge numerical and categorical sets
x_train = np.hstack((num_train_data, vec_train_data))
x_test = np.hstack((num_test_data, vec_test_data))

print np.shape(x_train)
print np.shape(x_test)

# rf = ensemble.RandomForestRegressor(n_estimators=200, max_depth=9)
reg = LinearRegression()


print "Cross validation is running..."
cv = cross_validation.KFold(len(y), n_folds=5, shuffle=True)

results = []
train_array = np.asanyarray(x_train)
target_array = np.asanyarray(y)
index = 1
for traincv, testcv in cv:
    probas = reg.fit(train_array[traincv], target_array[traincv]).predict(train_array[testcv])
    gini_score = normalized_gini(target_array[testcv], probas)
    print "GINI score for fold {}: {}".format(index, gini_score)
    index += 1
    results.append(gini_score)

# now print out the mean of the cross-validated results
print "Avg gini across all folds: " + str(np.array(results).mean())


# Build the submission file
reg.fit(x_train, y)
pred = reg.predict(x_test)

preds = pd.DataFrame({"Id": ids, "Hazard": pred})
preds = preds[['Id', 'Hazard']]
preds.to_csv('submit.csv', index=False)
print "\nSubmission file written!"