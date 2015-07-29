__author__ = 'dario'

import pandas as pd
import matplotlib.pyplot as plt
#import DataTransformer as dt
import numpy as np
from scipy.stats.stats import pearsonr

def revenue_analysis(filtered_data):
    sum_value = sum(filtered_data['Customer_Value'])
    revenue_per_customer = sum_value / len(filtered_data)
    print "Avg Revenue per customer is {}, Tot customer: {} Total revenue {}".format(revenue_per_customer, len(filtered_data), sum_value)

if __name__ == "__main__":

    # read the data in
    df = pd.read_csv("../../../resources/sclCustomerValue/modellingProjectTraining.txt", sep=";")

    #dt.show_some_stats(df)

    train_cols = df.columns[3:]
    target_column = df.columns[1]

    customer_value = df['Customer_Value']
    income = df['Income']
    age = df['Age']

    train = df['Age'].values

    binned_df = np.histogram(train, bins=5)
    print binned_df


    print "Sub set rows and cols: {}".format(df.shape)
    #df_valuable_customers_gender_m = df[(df.Customer_Value >= 500) & (df.Gender == 'M')]
    #df_valuable_customers_gender_f = df[(df.Customer_Value >= 500) & (df.Gender == 'F')]
    df_income_more_125000 = df[(df.Income >= 125000)].reset_index(drop=True)
    df_income_less_125000 = df[(df.Income < 125000)].reset_index(drop=True)

    #df.loc[df.Customer_Value < 500, 'Customer_Value'] = 'Low'
    #df.loc[df.Customer_Value >= 500, 'Customer_Value'] = 'High'


    customer_value_threshold = 300
    low_ = df[(df.Customer_Value < customer_value_threshold)].reset_index(drop=True)
    high_ = df[(df.Customer_Value >= customer_value_threshold)].reset_index(drop=True)

    '''
    filtered_data = df[
        (df.CE_Foreign_Travel_Model > 5) &
        #(df.CE_Physical_Fitness_Model < -1) &
        #(df.CE_Higher_Education_Model > 4) &
        #(df.CE_Higher_Education_Model < -4) &
        (df.CE_Avid_TV_Model < -6)
         & (df.CE_Social_Networking_Model < -6)
    ].reset_index(drop=True)
    '''

    filtered_data = df[
        (df.CE_Pet_Model > 4.5) &
        (df.CE_Donations_Model > 4) &
        (df.CE_Foreign_Travel_Model > 4) &
        (df.CE_Higher_Education_Model > 3) &
        (df.CE_Avid_TV_Model < -4)
    ].reset_index(drop=True)


    revenue_analysis(filtered_data)
    #revenue_analysis(df)


    fig_customer_value = plt.figure()
    fig_customer_value.suptitle('Customer value distribution', fontsize=14, fontweight='bold')
    ax1b = fig_customer_value.add_subplot(2, 1, 1)
    ax2b = fig_customer_value.add_subplot(2, 1, 2)

    ax1b.hist(filtered_data['Customer_Value'], bins=20, alpha=0.9)
    ax1b.set_xlabel('Customer_Value')

    ax2b.hist(df['Customer_Value'], bins=20, alpha=0.9)
    ax2b.set_xlabel("Customer_Value")

    plt.show()




    for column in train_cols:
        min_value = min(df[column])
        max_value = max(df[column])
        displacement = (max_value - min_value) / 3
        print "{} min: {}".format(column, min_value)
        print "{} max: {}".format(column, max_value)

        fig0 = plt.figure()
        fig0.suptitle('Feature ' + column, fontsize=14, fontweight='bold')
        ax1a = fig0.add_subplot(3, 1, 1)
        ax2a = fig0.add_subplot(3, 1, 2)
        ax3a = fig0.add_subplot(3, 1, 3)


        ax1a.hist(high_[column], bins=20, label="High", alpha=0.4)
        ax1a.hist(low_[column], bins=20, label="Low", alpha=0.4)
        ax1a.set_xlabel(column)
        ax1a.legend(loc='upper right')

        ax2a.hist(high_[column], bins=20, label="High", alpha=0.9)
        ax2a.set_xlabel(column + " High value")

        ax3a.hist(low_[column], bins=20, label="Low", alpha=0.9)
        ax3a.set_xlabel(column + " Low value")
        plt.show()






    print "Sub set rows and cols: {}".format(df_income_more_125000.shape)
    print "Sub set rows and cols: {}".format(df_income_less_125000.shape)


    #overlay example
    plt.hist((df_income_less_125000['Customer_Value']), bins=20, label="Less")
    plt.hist((df_income_more_125000['Customer_Value']), bins=20, label="More")
    plt.legend(loc='upper right')
    plt.show()

    print "First chart plotted!"


    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.hist(customer_value, bins=20, label="customer value")
    ax1.set_xlabel('Customer value')
    ax1.set_ylabel('Frequency')

    ax2.hist(age, bins=20, label="age")
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Frequency')

    ax3.hist(income, bins=20)
    ax3.set_xlabel('Income')
    ax3.set_ylabel('Frequency')
    plt.show()



    for column in train_cols:
        column_values = df[column]
        correlation = pearsonr(column_values, customer_value)
        print "Correlation for column {} is: {}".format(column, correlation)