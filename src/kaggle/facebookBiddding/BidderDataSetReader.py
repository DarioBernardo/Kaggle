__author__ = 'dario'



import pandas as pd


def main():
    pass


def readTrainData():

    df = pd.read_csv("../../../resources/facebookBidding/train_dataset.csv", sep=",")
    print "Data frame shape: {}".format(df.shape)

    target_column_name = 'outcome'

    full_target = df[target_column_name]

    df.drop(target_column_name, axis=1, inplace=True)

    return df, full_target


def readTestData():
    df = pd.read_csv("../../../resources/facebookBidding/test_dataset.csv", sep=",")
    print "Data frame shape: {}".format(df.shape)

    ids = df.bidder_id
    df.drop('bidder_id', axis=1, inplace=True)

    return df, ids
