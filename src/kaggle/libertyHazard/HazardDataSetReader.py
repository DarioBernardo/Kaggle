__author__ = 'dario'


import pandas as pd


def main():
    pass


def readTrainData():

    df = pd.read_csv("/home/dario/LogicalGlueDatasets/Liberty_Mutual_group_hazard_prediction/train.csv", sep=",")
    print "Data frame shape: {}".format(df.shape)

    target_column_name = 'Hazard'

    full_target = df[target_column_name]
    df.drop([target_column_name, 'Id'], axis=1, inplace=True)

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

    return full_data_set, full_target


def readTestData():

    df = pd.read_csv("/home/dario/LogicalGlueDatasets/Liberty_Mutual_group_hazard_prediction/test.csv", sep=",", index_col=0)
    print "Data frame shape: {}".format(df.shape)

    ids = df.index

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

    return full_data_set, ids

if __name__ == "__main__":
    print "Hazard dataset reader class"