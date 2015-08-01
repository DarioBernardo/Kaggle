__author__ = 'dario'



from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.preprocessing import Imputer

import pandas as pd

def build_classifiers():
    ESTIMATORS = {
        #"sgd": SGDClassifier(loss="modified_huber", penalty="elasticnet", n_iter=20000, class_weight='auto', alpha=0.1, epsilon=0.01),
        #"Extra trees": ExtraTreesClassifier(n_estimators=3000, max_depth=None, min_samples_leaf=1, random_state=0),
        "Logistic regression": LogisticRegression(),
        "Random forest": RandomForestClassifier(n_estimators=400, n_jobs=-1),
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
size = int(len(df) * 0.1)
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
top_correlated_features = corr.index[[range(0,40)]].values
train = train[top_correlated_features]

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
else :
    print "not performed!"





print "replacing NaNs..."
#Replace the NaNs with the mean value in column
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
cleaned_data = pd.DataFrame(imp.fit_transform(train), columns=train.columns)

print "Data-set shape: {}".format(cleaned_data.shape)

print "building random forest..."
rf = RandomForestRegressor(n_estimators=400, verbose=True, n_jobs=-1)
rf.fit(cleaned_data, target_d)

importance = rf.feature_importances_
feature_ranking = pd.Series(importance, index=cleaned_data.columns)
feature_ranking.sort(ascending=False)
print "Feature importance:"
'''
for index in range(0, len(importance)):
    print("{} -> {}".format(column_names[index], importance[index]))
'''

print feature_ranking

# Reduce the feature to 30
final_selected_features = feature_ranking.index[[range(0,30)]].values

final_train_data =  np.array(cleaned_data[final_selected_features])
target_b = np.array(target_b)

print "Data-set shape: {}".format(final_train_data.shape)


# Binary classification
estimators = build_classifiers()
cv = cross_validation.StratifiedKFold(target_b, n_folds=5)
results = []
index = 1
for train_indices, test_indices in cv:
    X_train, X_test = final_train_data[train_indices], final_train_data[test_indices]
    y_train, y_test = target_b[train_indices], target_b[test_indices]

    model_scores = []
    models_used = []
    for name, estimator in estimators.items():
        models_used.append(name)
        print "I am trying to build a {}".format(name)
        probas = estimator.fit(X_train, y_train).predict_proba(X_test)
        auc_score = roc_auc_score(y_test, [x[1] for x in probas])
        print "AUC score for fold {}: {}".format(index, auc_score)
        index += 1
        model_scores.append(auc_score)

    results_on_fold = pd.Series(model_scores, index=models_used)
    print "AUC score for fold {}: {}".format(index, results_on_fold)
    index += 1
    results.append(results_on_fold)

# now print out the mean of the cross-validated results
print "All results: {}".format(results)
print results