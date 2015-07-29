__author__ = 'dario'

def main():
    pass

def print_columns(data):
    print "\nCOLUMNS IN DATA SET ARE:"
    for x in data.columns:
        print x

def append_squared_value_for_columns(data, columns_to_square):
    added_column_names = []
    for column in columns_to_square:
        squared = data[column] * data[column]
        new_column_name = '{}_squared'.format(column)
        added_column_names.append(new_column_name)
        data[new_column_name] = squared # append the squared value of the column

    #print_columns(data)
    return added_column_names


def append_cubed_value_for_columns(data, columns_to_cube):
    added_column_names = []
    for column in columns_to_cube:
        squared = data[column] * data[column] * data[column]
        new_column_name = '{}_cubed'.format(column)
        added_column_names.append(new_column_name)
        data[new_column_name] = squared # append the squared value of the column

    #print_columns(data)
    return added_column_names


def append_specific_combined_features(data, features_src, features_dest):
    for column in features_src:
        for other_column in features_dest:
            combined_column = data[column] * data[other_column]
            data["{}_and_{}_comb".format(column, other_column)] = combined_column


def append_features_combined_with_each_other(data, features_to_combine):
    for column_index in range(0, len(features_to_combine)):
        column = features_to_combine[column_index]
        for other_column_index in range(0, len(features_to_combine)):
            if column_index < other_column_index:
                other_column = features_to_combine[other_column_index]
                combined_column = data[column] * data[other_column]
                data["{}_and_{}_comb".format(column, other_column)] = combined_column # append the squared value of the column

                #print_columns(data)

def show_some_stats(df):
    # take a look at the dataset
    # It prints the first 5 rows
    print df.head()

    print "\nCOLUMNS:"
    for x in df.columns:
        print x

    # summarize the data
    print "\nDATA SUMMARY:"
    print df.describe()

    # take a look at the standard deviation of each column
    print "\nVARIABLES STDDEV:"
    print df.std()


if __name__ == "__main__":
    main()