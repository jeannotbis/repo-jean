import lz4.frame
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator

#I define the paths of the datasets

path_train = 'data/train.lz4'
path_test = 'data/test.lz4'

#I define the types of the variables

numerical = ['demand','od_destination_time','od_number_of_similar_12_hours','od_number_of_similar_2_hours',
             'od_number_of_similar_4_hours','od_origin_time','od_travel_time_minutes',
             'price','sale_day_x','origin_days_to_next_public_holiday','origin_days_to_next_school_holiday',
             'destination_days_to_next_public_holiday','destination_days_to_next_school_holiday','sale_day']

date = ['departure_date','sale_date']

categorical = ['dataset_type','destination_station_name','origin_station_name','sale_month','sale_week','sale_weekday','sale_year',
             'origin_current_public_holiday','origin_current_school_holiday','od_origin_week','od_origin_weekday','od_origin_year','od_origin_month',
             'destination_current_public_holiday','destination_current_school_holiday']

def initialize_dataframe(path) :
    with lz4.frame.open(path,'r') as file :
        bytes = file.read()
        s = str(bytes)[2:]
    df = pd.DataFrame([x.split(',') for x in s.split('\\r\\n')])
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df = df.dropna()
    df = df.convert_dtypes()
    df[numerical] = df[numerical].apply(pd.to_numeric)
    return(df)

df_train = initialize_dataframe(path_train)
df_test = initialize_dataframe(path_test)

def first_look_at_the_datasets() :
    print(df_train.head)
    print(df_test.head)
    print(df_train.info())
    print(df_train.iloc[1])
    print(df_train.iloc[100])
    print(df_train[numerical].max(axis=0))
    print(df_train[numerical].min(axis=0))
    return()

"""first_look_at_the_datasets()"""

def histogram_plot() :
    for column_name in numerical :
        hist = df_train[[column_name]].hist(bins = 25)
        plt.savefig('histograms_numerical/'+column_name+'_histogram.png')
        plt.clf()
    for column_name in categorical :
        plot = sns.countplot(df_train[column_name]).set_title('Histogram')
        plt.savefig('histograms_categorical/'+column_name+'_histogram.png')
        plt.clf()
    return()

"""histogram_plot()"""


#I suppress the rows for whici od_number_of_similar_12_hours is equal to -1 (which means that its value is unknown)
df_train = df_train.drop(df_train[df_train.od_number_of_similar_12_hours == -1].index, axis = 0)
df_test = df_test.drop(df_test[df_test.od_number_of_similar_12_hours == -1].index, axis = 0)

#I suppress the column destination_current_public_holiday
categorical.remove('destination_current_public_holiday')
df_train = df_train.drop(columns = ['destination_current_public_holiday'], axis=1)
df_test = df_test.drop(columns = ['destination_current_public_holiday'], axis=1)

#I suppress the column origin_days_to_next_public_holiday
numerical.remove('origin_days_to_next_public_holiday')
df_train = df_train.drop(columns = ['origin_days_to_next_public_holiday'], axis=1)
df_test = df_test.drop(columns = ['origin_days_to_next_public_holiday'], axis=1)


#print(df_train.loc[df_train['sale_week']!=df_train['sale_weekday']].shape)
#Executing the line of code above shows that sale_week is everywhere equal to sale_week_day
#Therefore I suppress the sale_week column from the datasets
categorical.remove('sale_week')
df_train = df_train.drop(columns = ['sale_week'], axis=1)
df_test = df_test.drop(columns = ['sale_week'], axis=1)

#I suppress the column dataset_type
categorical.remove('dataset_type')
df_train = df_train.drop(columns = ['dataset_type'], axis=1)
df_test = df_test.drop(columns = ['dataset_type'], axis=1)


#In order to split df_train into a new train set and a validation set, I need to sort it by sale_date.
#This is because in order to have a meaningful validation method, I must train my model on the past 
# and validate its performance on datapoints from the futur.
df_train_sorted = df_train.sort_values(by=['sale_date'])
#Splitting df_train into a training and a validation dataframe
size_new_train_set = 500000
df_train_partial = df_train_sorted[:size_new_train_set]
df_valid = df_train_sorted[size_new_train_set:]

def scatter_plot() :
    for column_name in numerical :
        if column_name != 'demand':
            scatter_plot = sns.scatterplot(x=df_train[column_name], y=df_train['demand']).set_title('Scatterplot demand versus '+column_name)
            plt.savefig('scatter_plots/'+'demand_vs_'+column_name+'_scatterplot.png')
            plt.clf()
    return()

scatter_plot()

def box_plot() :
    for column_name in categorical :
        if column_name != 'dataset_type':
            sorted_nb = df_train.groupby([column_name])['demand'].median().sort_values()
            box = sns.boxplot(x=df_train[column_name], y=df_train['demand'], order=list(sorted_nb.index))
            if column_name == 'od_origin_week' :
                box.tick_params(labelsize=6)
            plt.savefig('box_plots/'+'demand_vs_'+column_name+'_boxplot.png')
            plt.clf()
    return()

"""box_plot()"""

h2o.init()

#Transform pandas dataframes into h2o frames
hf_train = h2o.H2OFrame(df_train)
hf_train_partial = h2o.H2OFrame(df_train_partial)
hf_valid = h2o.H2OFrame(df_valid)
hf_test = h2o.H2OFrame(df_test)


#Turning categorical variables into "factor" for h2o to process them as categorical variables
hf_train[categorical] = hf_train[categorical].asfactor()
hf_train_partial[categorical] = hf_train_partial[categorical].asfactor()
hf_valid[categorical] = hf_valid[categorical].asfactor()
hf_test[categorical] = hf_test[categorical].asfactor()

predictors = (numerical + categorical).remove('demand')
response = 'demand'



def evaluate_model(df):
    #not allowing predicted demand to be negative
    df['prediction'] = df['prediction'].clip(0, None)
    #not allowing values to be NaN
    df = df.dropna()
    #list of sale_day_x for which I will compute the cumulative demand
    list_of_sale_day_x = [-90,-60,-30,-20,-15,-10,-7,-6,-5,-3,-2,-1]
    #first of all I compute a table with the average absolute and relative error on the cumulative predicted demand for different values of sale_day_x
    for min_sale_day_x in list_of_sale_day_x :
        df_new = df.loc[df['sale_day_x'] >= min_sale_day_x]
        df_groupby = df_new.groupby(['departure_date','od_origin_time','origin_station_name','destination_station_name'], as_index = False)
        df_cumulative_demand = df_groupby['demand'].sum()
        df_cumulative_prediction = df_groupby['prediction'].sum()
        df_join = pd.merge(df_cumulative_demand, df_cumulative_prediction, on = ['departure_date','origin_station_name','od_origin_time','destination_station_name'])
        df_join['absolute_error'] = (df_join['demand']-df_join['prediction']).abs()
        df_join['relative_error'] = (df_join['demand']-df_join['prediction']).abs()/df_join['demand']
        df_join_groupby = df_join.groupby(['origin_station_name','destination_station_name'], as_index=False)
        df_average_absolute = df_join_groupby['absolute_error'].mean()
        #Computing the relative error can result in division by zero, therefore in this case I must remove Nan and inf values before groupby
        df_join = df_join.replace([np.inf, -np.inf], np.nan).dropna()
        df_join_groupby = df_join.groupby(['origin_station_name','destination_station_name'], as_index=False)
        df_average_relative = df_join_groupby['relative_error'].mean()
        df_average_results = pd.merge(df_average_absolute, df_average_relative, on = ['origin_station_name','destination_station_name'])
        print('\n')
        print('TABLE OF CUMULATIVE ERROR PER ORIGIN/DESTINATION FOR SALE_DAY_X = '+str(min_sale_day_x))
        print('\n')
        print(df_average_results)
    #then I compute a custom metric which is the average absolute error on the predicted demand times the price of the ticket, divided by
    #the average price of a ticket
    #I compute it for each origin/destination which I display in a table
    df['custom metric'] = (df['demand'] - df['prediction']).abs()*df['price']
    df_groupby = df.groupby(['origin_station_name','destination_station_name'], as_index = False)
    df_origin_destination_custom_metric = df_groupby['custom metric'].mean()
    print('\n')
    print('TABLE OF CUSTOM METRIC PER ORIGIN/DESTINATION')
    print('\n')
    print(df_origin_destination_custom_metric)
    #Now I compute the custom metric on the whole dataset
    custom_metric = df['custom metric'].mean()
    print('\n')
    print('CUSTOM METRIC = '+str(custom_metric))
    print('\n')
    return()



#This function can execute different models in order to choose one of them based on its results on the validation set
def train_model_and_compute_results_on_validation_set():
    #build and train the model:
    """demand_drf = H2ORandomForestEstimator(ntrees=100,
                                        max_depth=8,
                                        min_rows=10,
                                        seed=1112
                                        )"""
    demand_drf = H2OGradientBoostingEstimator(seed=1111,
                                            ntrees=100,
                                            max_depth=5)
    demand_drf.train(x=predictors,
                y=response,
                training_frame=hf_train_partial)
    #eval performance on partial training set
    print("\n")
    print("PERFORMANCES ON HF_TRAIN_PARTIAL :")
    perf_train = demand_drf.model_performance()
    print(perf_train)
    #eval performance on validation set
    print("PERFORMANCES ON HF_VALID :")
    perf_valid = demand_drf.model_performance(hf_valid)
    print(perf_valid)
    pred_valid = demand_drf.predict(hf_valid)
    pred_valid_df = h2o.as_list(pred_valid)
    df_valid.insert(2,'prediction',pred_valid_df,True)
    print(df_valid.loc[:,['prediction','demand','od_origin_time']])
    evaluate_model(df_valid)
    return(demand_drf)


""" train_model_and_compute_results_on_validation_set() """



#This function trains the selected model on the full training set and computes the final results on the test set
def train_selected_model_and_compute_results_on_test_set():
    demand_drf = H2OGradientBoostingEstimator(seed=1111,
                                            ntrees=100,
                                            max_depth=5)
    demand_drf.train(x=predictors,
                y=response,
                training_frame=hf_train)
    #eval performance on full training set
    print("\n")
    print("PERFORMANCES ON HF_TRAIN :")
    perf_train = demand_drf.model_performance()
    print(perf_train)
    #eval performance on test set
    print("PERFORMANCES ON HF_TEST :")
    perf_test = demand_drf.model_performance(hf_test)
    print(perf_test)
    pred_test = demand_drf.predict(hf_test)
    pred_test_df = h2o.as_list(pred_test)
    df_test.insert(2,'prediction',pred_test_df,True)
    evaluate_model(df_test)
    return()


train_selected_model_and_compute_results_on_test_set()



