__author__ = 'dario'

import pandas as pd
import numpy as np
import random
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import HazardDataSetReader as dr

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

hidden_size = 100
epochs = 6

#load train and test data
train, labels = dr.readTrainData()
test, test_ids = dr.readTestData()

assert len(train.columns) == len(test.columns)


columns = train.columns

size = int(len(train) * 0.8)
rows = random.sample(train.index, size)

x_train = np.array(train)
x_test = np.array(test)
y_train = np.array(labels)
y_train = y_train.reshape(len(y_train), 1)

input_size = x_train.shape[1]
target_size = y_train.shape[1]


# prepare dataset

ds = SDS( input_size, target_size )
ds.setField('input', x_train )
ds.setField('target', y_train )

# init and train

net = buildNetwork( input_size, hidden_size, target_size, bias = True )
trainer = BackpropTrainer(net, ds, learningrate=0.0001, momentum=0.001, verbose=True, weightdecay=0.01)

print "training for {} epochs...".format( epochs )

for i in range( epochs ):
    mse = trainer.train()
    #mse = trainer.trainEpochs(10)
    rmse = sqrt( mse )
    print "training RMSE, epoch {}: {}".format(i + 1, rmse)

preds_train = []
for x in x_train:
    preds_train.append(net.activate(x))

gini_score = normalized_gini(labels, preds_train)
print "GINI score: {}".format(gini_score)

preds = []
for x in x_test:
    preds.append(net.activate(x))

preds = np.array(preds).reshape(-1)
preds = pd.DataFrame({"Id": test_ids, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('nn_benchmark.csv')

print "Done!"
#pickle.dump( net, open( output_model_file, 'wb' ))
