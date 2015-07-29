__author__ = 'dario'

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
from numpy import genfromtxt, savetxt
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

def discretize_array(predicted_probs):
    discretizedPrediction = [None] * len(predicted_probs)
    for index in range (0, len(predicted_probs)):
        probability = predicted_probs[index][1]
        if probability > 0.5:
            discretizedPrediction[index] = 1
        else:
            discretizedPrediction[index] = 0

    return discretizedPrediction

def show_errors(train, rf):
    for index in range(0, len(train)):
        proba = rf.predict_proba(train[index])
    prediction = 0
    if(proba[0][1] > 0.5):
        prediction = 1

    if(prediction != target[index]):
        print "ERROR AT INDEX {}, OUT CLASS SHOULD BE {} :".format(index, target[index])
        print train[index]

print("Running")

trainFilename = '../../../resources/facebookBidding/train_dataset.csv'
file_io = open(trainFilename, 'r')
csvData = csv.reader(file_io, delimiter=',')
header = csvData.next()
file_io.close()

print("\nHeader = {}\n".format(header))


#create the training & test sets, skipping the header row with [1:]
dataSet = genfromtxt(open(trainFilename, 'r'), delimiter=',', dtype='f8')[1:]
print("\nData dimension = {}\n".format(dataSet.shape))

#clean_data(dataSet)


#print(dataset)
target = [x[len(x)-1] for x in dataSet]
train = [x[:-1] for x in dataSet]


train_row_number = len(train)
if train_row_number != 1984:
    print "\n\n\n SOMETHING IS TOTALLY WRONG!! THERE SHOULD BE 1984 ROWS \n\n"
print("\nTrain rows = {}\n".format(train_row_number))
print("\nTrain columns = {}\n".format(len(train[0])))

print("\nTarget rows = {}\n".format(len(target)))
#print(target)

rf = RandomForestClassifier(n_estimators=8000) # original value 1200
#rf.fit(train, target)

cv = cross_validation.StratifiedKFold(target, n_folds=5)

#iterate through the training and test cross validation segments and
#run the classifier on each one, aggregating the results into a list
print "Cross validation is running..."
results = []
train_array = np.asanyarray(train)
target_array = np.asanyarray(target)
index = 1
for traincv, testcv in cv:
    probas = rf.fit(train_array[traincv], target_array[traincv]).predict_proba(train_array[testcv])
    auc_score = roc_auc_score(target_array[testcv], [x[1] for x in probas])
    print "AUC score for fold {}: {}".format(index, auc_score)
    index += 1
    results.append(auc_score)

# now print out the mean of the cross-validated results
print "Avg auc across all folds: " + str(np.array(results).mean())

#show_errors(train, rf)


predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(train))]

print("\npredicted_probs rows = {}\n".format(len(predicted_probs)))
print("\npredicted_probs columns = {}\n".format(len(predicted_probs[0])))

'''
importance = rf.feature_importances_
print "Feature importance:"
for index in range(0, len(importance)):
    print("{} -> {}".format(header[index], importance[index]))
'''

discretizedPrediction = discretize_array(predicted_probs)

cm = confusion_matrix(target, discretizedPrediction)
print "Confusion matrix on training:"
print(cm)


# OK NOW LET'S GET THE TEST DATA
rf.fit(train, target)  #I need to train on the whole training set, not only on a single fold, before making the predictions on the test set.
testFilename = 'facebookBidding/test_dataset.csv'

testDataSet = genfromtxt(open(testFilename, 'r'), delimiter=',', dtype='f8')[1:]
print("\nTest Data dimension = {}\n".format(testDataSet.shape))

test = [x[1:] for x in testDataSet] # getting rid of the bidder_id

# Reopen the file with a proper CSV parser and read only the bidder_id (otherwhise i get nan only from the previous method)
testBidderId = [None] * len(test)
index = 0
with open(testFilename, 'rb') as csvfile:
    testDataForIds = csv.reader(csvfile, delimiter=',')
    testDataForIds.next() # get rid of the header
    for row in testDataForIds:
        testBidderId[index] = row[0]
        index += 1


importance = rf.feature_importances_
print "Feature importance:"
for index in range(0, len(importance)):
    print("{} -> {}".format(header[index], importance[index]))


test_predicted_probs = [[testBidderId[index], x[1]] for index, x in enumerate(rf.predict_proba(test))]

submission = [[testBidderId[index], test_predicted_probs[index][1]] for index in range(0, len(test_predicted_probs))]

savetxt('submission_last.csv', submission, delimiter=',', fmt='%s,%s', header='bidder_id,prediction', comments='')


predicted_probs_only = [[predicted_probs[index][1]] for index in range(0, len(predicted_probs))]

# overall accuracy
acc = rf.score(train,target)
print("\nAccuracy = {}\n".format(acc))


print "Finish!"

'''
# get roc/auc info
fpr = dict()
tpr = dict()
fpr, tpr, _ = roc_curve(target, predicted_probs_only)

roc_auc = dict()
roc_auc = auc(fpr, tpr)

# make the plot
plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
plt.legend(loc="lower right", shadow=True, fancybox=True)
plt.show()

'''


