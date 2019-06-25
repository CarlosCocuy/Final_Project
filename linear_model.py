import csv
import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import time

from datetime import date, datetime, time, timedelta
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings("ignore")
count=1
#### Input params ##################
test_size = 0.2                 # proportion of dataset to be used as test set
cv_size = 0.2                   # proportion of dataset to be used as cross-validation set
Nmax = 10                       # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
                                # Nmax is the maximum N we are going to test
fontsize = 14
ticklabelsize = 14
####################################
field_names=['ticker','RMSE','R2', 'MAPE', 'N_OPT']
with open('lineregOut.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(field_names)
def get_preds_lin_reg(d, target_col, N, pred_min, offset):
    """
    Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe with the values you want to predict. Can be of any length.
        target_col : name of the column you want to predict e.g. 'adj_close'
        N          : get prediction at timestep t using values from t-1, t-2, ..., t-N
        pred_min   : all predictions should be >= pred_min
        offset     : for df we only do predictions for df[offset:]. e.g. offset can be size of training set
    Outputs
        pred_list  : the predictions for target_col. np.array of length len(df)-offset.
    """
    # Create linear regression object
    regr = LinearRegression(fit_intercept=True)

    pred_list = []
    for i in range(offset, len(d['adj_close'])):
        X_train = np.array(range(len(d['adj_close'][i-N:i]))) # e.g. [0 1 2 3 4]

        y_train = np.array(d['adj_close'][i-N:i]) # e.g. [2944 3088 3226 3335 3436]
        X_train = X_train.reshape(-1, 1)

        y_train = y_train.reshape(-1, 1)

        regr.fit(X_train, y_train)            # Train the model
        Narray = [[N]]
        pred = regr.predict(Narray)

        pred_list.append(pred[0][0])  # Predict the footfall using the model

    # If the values are < pred_min, set it to be pred_min
    pred_list = np.array(pred_list)
    pred_list[pred_list < pred_min] = pred_min

    return pred_list

def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import os

directory_in_str = 'fang'

directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print(count,filename)
        df = pd.read_csv(directory_in_str + '/' +filename, sep = ",")


        # Convert Date column to datetime
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')

        # Change all column headings to be lower case, and remove spacing
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

        # Sort by datetime
        df.sort_values(by='date', inplace=True, ascending=True)

        # Get sizes of each of the datasets
        num_cv = int(cv_size*len(df))
        num_test = int(test_size*len(df))
        num_train = len(df) - num_cv - num_test

        # Split into train, cv, and test
        train = df[:num_train]
        cv = df[num_train:num_train+num_cv]
        train_cv = df[:num_train+num_cv]
        test = df[num_train+num_cv:]

        # Plot adjusted close over time

        RMSE = []

        for N in range(2, Nmax+1): # N is no. of samples to use to predict the next value
            est_list = get_preds_lin_reg(train_cv, 'adj_close', N, 0, num_train)

            cv['est' + '_N' + str(N)] = est_list
            RMSE.append(math.sqrt(mean_squared_error(est_list, cv['adj_close'])))

        # Set optimum N
        N_opt = np.argmin(RMSE) +2

        day = pd.Timestamp(date(2017, 10, 31))
        df_temp = cv[cv['date'] <= day]
        regr = LinearRegression(fit_intercept=True) # Create linear regression object
        # Plot the linear regression lines
        X_train = np.array(range(len(df_temp['adj_close'][-N_opt-1:-1]))) # e.g. [0 1 2 3 4]
        y_train = np.array(df_temp['adj_close'][-N_opt-1:-1]) # e.g. [2944 3088 3226 3335 3436]
        X_train = X_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        regr.fit(X_train, y_train)            # Train the model
        y_est = regr.predict(X_train)

        est_list = get_preds_lin_reg(df, 'adj_close', N_opt, 0, num_train+num_cv)

        test['est' + '_N' + str(N_opt)] = est_list
        RMSE = math.sqrt(mean_squared_error(est_list, test['adj_close']))
        R2 = r2_score(test['adj_close'], est_list)
        MAPE = get_mape(test['adj_close'], est_list)

        ticker = filename.replace('.csv', '')
        # ticker = ticker.replace('splitdata/','')
        count=count+1

        fields=[ticker,RMSE,R2, MAPE, N_opt]
        with open('lineregOut.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        continue
    else:
        continue
