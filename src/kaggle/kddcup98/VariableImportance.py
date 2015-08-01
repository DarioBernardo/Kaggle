from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

__author__ = 'dario'



import csv
import random
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.preprocessing import Imputer

import pandas as pd

def build_classifiers():
    ESTIMATORS = {
        #"sgd": SGDClassifier(loss="modified_huber", penalty="elasticnet", n_iter=20000, class_weight='auto', alpha=0.1, epsilon=0.01),
        #"Extra trees": ExtraTreesClassifier(n_estimators=3000, max_depth=None, min_samples_leaf=1, random_state=0),
        "Logistic regression": LogisticRegression(),
        "Random forest": RandomForestRegressor(n_estimators=400, n_jobs=-1),
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


# I throw away "OSOURCE", "ZIP", "RFA_2R" and "RFA_23" as they contains to many categorical values and the number of columns may explode after dummification
columns_to_use = [
    "ODATEDW", "TCODE", "STATE", "MAILCODE", "PVASTATE", "DOB", "NOEXCH", "RECINHSE", "RECP3", "RECPGVG", "RECSWEEP",
    "MDMAUD", "DOMAIN", "CLUSTER", "AGE", "AGEFLAG", "HOMEOWNR", "CHILD03", "CHILD07", "CHILD12", "CHILD18", "NUMCHLD",
    "INCOME", "GENDER", "WEALTH1", "HIT", "MBCRAFT", "MBGARDEN", "MBBOOKS", "MBCOLECT", "MAGFAML", "MAGFEM", "MAGMALE",
    "PUBGARDN", "PUBCULIN", "PUBHLTH", "PUBDOITY", "PUBNEWFN", "PUBPHOTO", "PUBOPP", "DATASRCE", "MALEMILI", "MALEVET",
    "VIETVETS", "WWIIVETS", "LOCALGOV", "STATEGOV", "FEDGOV", "SOLP3", "SOLIH", "MAJOR", "WEALTH2", "GEOCODE", "COLLECT1",
    "VETERANS", "BIBLE", "CATLG", "HOMEE", "PETS", "CDPLAY", "STEREO", "PCOWNERS", "PHOTO", "CRAFTS", "FISHER", "GARDENIN",
    "BOATS", "WALKER", "KIDSTUFF", "CARDS", "PLATES", "LIFESRC", "PEPSTRFL", "POP901", "POP902", "POP903", "POP90C1",
    "POP90C2", "POP90C3", "POP90C4", "POP90C5", "ETH1", "ETH2", "ETH3", "ETH4", "ETH5", "ETH6", "ETH7", "ETH8", "ETH9",
    "ETH10", "ETH11", "ETH12", "ETH13", "ETH14", "ETH15", "ETH16", "AGE901", "AGE902", "AGE903", "AGE904", "AGE905",
    "AGE906", "AGE907", "CHIL1", "CHIL2", "CHIL3", "AGEC1", "AGEC2", "AGEC3", "AGEC4", "AGEC5", "AGEC6", "AGEC7",
    "CHILC1", "CHILC2", "CHILC3", "CHILC4", "CHILC5", "HHAGE1", "HHAGE2", "HHAGE3", "HHN1", "HHN2", "HHN3", "HHN4",
    "HHN5", "HHN6", "MARR1", "MARR2", "MARR3", "MARR4", "HHP1", "HHP2", "DW1", "DW2", "DW3", "DW4", "DW5", "DW6", "DW7",
    "DW8", "DW9", "HV1", "HV2", "HV3", "HV4", "HU1", "HU2", "HU3", "HU4", "HU5", "HHD1", "HHD2", "HHD3", "HHD4", "HHD5",
    "HHD6", "HHD7", "HHD8", "HHD9", "HHD10", "HHD11", "HHD12", "ETHC1", "ETHC2", "ETHC3", "ETHC4", "ETHC5", "ETHC6",
    "HVP1", "HVP2", "HVP3", "HVP4", "HVP5", "HVP6", "HUR1", "HUR2", "RHP1", "RHP2", "RHP3", "RHP4", "HUPA1", "HUPA2",
    "HUPA3", "HUPA4", "HUPA5", "HUPA6", "HUPA7", "RP1", "RP2", "RP3", "RP4", "MSA", "ADI", "DMA", "IC1", "IC2", "IC3",
    "IC4", "IC5", "IC6", "IC7", "IC8", "IC9", "IC10", "IC11", "IC12", "IC13", "IC14", "IC15", "IC16", "IC17", "IC18",
    "IC19", "IC20", "IC21", "IC22", "IC23", "HHAS1", "HHAS2", "HHAS3", "HHAS4", "MC1", "MC2", "MC3", "TPE1", "TPE2",
    "TPE3", "TPE4", "TPE5", "TPE6", "TPE7", "TPE8", "TPE9", "PEC1", "PEC2", "TPE10", "TPE11", "TPE12", "TPE13", "LFC1",
    "LFC2", "LFC3", "LFC4", "LFC5", "LFC6", "LFC7", "LFC8", "LFC9", "LFC10", "OCC1", "OCC2", "OCC3", "OCC4", "OCC5",
    "OCC6", "OCC7", "OCC8", "OCC9", "OCC10", "OCC11", "OCC12", "OCC13", "EIC1", "EIC2", "EIC3", "EIC4", "EIC5", "EIC6",
    "EIC7", "EIC8", "EIC9", "EIC10", "EIC11", "EIC12", "EIC13", "EIC14", "EIC15", "EIC16", "OEDC1", "OEDC2", "OEDC3",
    "OEDC4", "OEDC5", "OEDC6", "OEDC7", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7", "EC8", "SEC1", "SEC2", "SEC3",
    "SEC4", "SEC5", "AFC1", "AFC2", "AFC3", "AFC4", "AFC5", "AFC6", "VC1", "VC2", "VC3", "VC4", "ANC1", "ANC2", "ANC3",
    "ANC4", "ANC5", "ANC6", "ANC7", "ANC8", "ANC9", "ANC10", "ANC11", "ANC12", "ANC13", "ANC14", "ANC15", "POBC1", "POBC2",
    "LSC1", "LSC2", "LSC3", "LSC4", "VOC1", "VOC2", "VOC3", "HC1", "HC2", "HC3", "HC4", "HC5", "HC6", "HC7", "HC8", "HC9",
    "HC10", "HC11", "HC12", "HC13", "HC14", "HC15", "HC16", "HC17", "HC18", "HC19", "HC20", "HC21", "MHUC1", "MHUC2", "AC1",
    "AC2", "ADATE_2", "ADATE_3", "ADATE_4", "ADATE_5", "ADATE_6", "ADATE_7", "ADATE_8", "ADATE_9", "ADATE_10", "ADATE_11",
    "ADATE_12", "ADATE_13", "ADATE_14", "ADATE_15", "ADATE_16", "ADATE_17", "ADATE_18", "ADATE_19", "ADATE_20", "ADATE_21",
    "ADATE_22", "ADATE_23", "ADATE_24", "RFA_2", "RFA_3", "RFA_4", "RFA_5", "RFA_6", "RFA_7", "RFA_8", "RFA_9", "RFA_10",
    "RFA_11", "RFA_12", "RFA_13", "RFA_14", "RFA_15", "RFA_16", "RFA_17", "RFA_18", "RFA_19", "RFA_20", "RFA_21", "RFA_22",
    "RFA_24", "CARDPROM", "MAXADATE", "NUMPROM", "CARDPM12", "NUMPRM12", "RDATE_3", "RDATE_4", "RDATE_5", "RDATE_6",
    "RDATE_7", "RDATE_8", "RDATE_9", "RDATE_10", "RDATE_11", "RDATE_12", "RDATE_13", "RDATE_14", "RDATE_15", "RDATE_16",
    "RDATE_17", "RDATE_18", "RDATE_19", "RDATE_20", "RDATE_21", "RDATE_22", "RDATE_23", "RDATE_24", "RAMNT_3", "RAMNT_4",
    "RAMNT_5", "RAMNT_6", "RAMNT_7", "RAMNT_8", "RAMNT_9", "RAMNT_10", "RAMNT_11", "RAMNT_12", "RAMNT_13", "RAMNT_14",
    "RAMNT_15", "RAMNT_16", "RAMNT_17", "RAMNT_18", "RAMNT_19", "RAMNT_20", "RAMNT_21", "RAMNT_22", "RAMNT_23", "RAMNT_24",
    "RAMNTALL", "NGIFTALL", "CARDGIFT", "MINRAMNT", "MINRDATE", "MAXRAMNT", "MAXRDATE", "LASTGIFT", "LASTDATE", "FISTDATE",
    "NEXTDATE", "TIMELAG", "AVGGIFT", "CONTROLN", "TARGET_B", "TARGET_D", "HPHONE_D", "RFA_2F", "RFA_2A", "MDMAUD_R",
    "MDMAUD_F", "MDMAUD_A", "CLUSTER2", "GEOCODE2"]

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

final_train_data = cleaned_data[final_selected_features]
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