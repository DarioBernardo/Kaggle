__author__ = 'dario'


import pandas as pd
import random
import DataTransformer as dt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr

def main():
    pass

def add_extra_features(data, original_column_name):
    # append a squared value for each column
    added_column_names = dt.append_squared_value_for_columns(data, original_column_name)
    # Append Features cubed
    dt.append_cubed_value_for_columns(data, original_column_name)
    # append a combination of all the features
    dt.append_features_combined_with_each_other(data, original_column_name)

    dt.append_features_combined_with_each_other(data, added_column_names[:len(added_column_names)-1])

if __name__ == "__main__":

    # read the data in
    df = pd.read_csv("../../../resources/sclCustomerValue/modellingProjectTraining.txt", sep=";")

    #show_some_stats(df)

    #df['gender'] = df['gender'].astype('category')  # Categorize!  

    train_cols = df.columns[3:]
    target_column = df.columns[1]
    # Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

    print "Training columns name"
    print train_cols

    print "Target column"
    print target_column

    train_full = df[train_cols]
    target_full = df[target_column]
    customer_ids = df[df.columns[0]]


    #plt.plot(customer_ids, target_full, 'ro')
    #plt.show()


    if (len(target_full)!=len(train_full)):
        raise Exception("Training set and target column size must be equal!")


    #ADD SOME EXTRA ENGINEERED FEATURES
    add_extra_features(train_full, train_cols)


    # Add gender column
    le_sex = preprocessing.LabelEncoder()
    train_full['gender'] = le_sex.fit_transform(df[df.columns[2]])
    #enc.fit(df[df.columns[2]])
    #train_full['gender'] = enc.transform()

    dt.print_columns(train_full)
    print train_full.head()

    #scaler = preprocessing.StandardScaler()
    #train_full_scaled = scaler.fit_transform(train_full)
    #new_columns = train_full.columns
    #train_full = pd.DataFrame(train_full_scaled, columns=new_columns)

    size = int(len(train_full) * 0.81)
    rows = random.sample(train_full.index, size)
    train = train_full.ix[rows]
    target = target_full.ix[rows]
    validation = train_full.drop(rows)
    validation_target = target_full.drop(rows)

    print("\nTrain rows = {}\n".format(len(train)))
    print("\nValidation rows = {}\n".format(len(validation)))
    print("\nNumber of columns = {}\n".format(len(train_full.columns)))


    rf = linear_model.LinearRegression()
    #rf = linear_model.SGDRegressor()

    rf.fit(train, target)
    predictions = rf.predict(validation)

    error = mean_squared_error(validation_target, predictions)
    #error = mean_squared_error(validation_target, [x[1] for x in predictions])
    correlation = pearsonr(validation_target, predictions)

    print ("\nError = {}\n".format(error))
    print ("\nPearson's correlation = {}\n".format(correlation))

    print "Done!"


    # Now building the test data!
    df_test = pd.read_csv("../../../resources/sclCustomerValue/modellingProjectModelling.txt", sep=";")

    test_full = df_test[train_cols]
    test_customer_ids = df_test[df.columns[0]]

    #ADD THE SAME EXTRA ENGINEERED FEATURES
    add_extra_features(test_full, train_cols)


    test_full['gender'] = le_sex.fit_transform(df_test[df_test.columns[1]])
    #enc.fit(df[df.columns[2]])
    #train_full['gender'] = enc.transform()

    dt.print_columns(test_full)
    print test_full.head()

    test_predictions = rf.predict(test_full)

    print ("\nTest prediction = {}\n".format(test_predictions))

    submission = [[test_customer_ids[index], test_predictions[index]] for index in range(0, len(test_predictions))]

    #savetxt('modelling_submission.txt', submission, delimiter=';', fmt='%s;%s', header='Customer_ID;Customer_Value', comments='')

    print ("\nPearson's correlation on validation was = {}\n".format(correlation))

    print "Done!"

    main()