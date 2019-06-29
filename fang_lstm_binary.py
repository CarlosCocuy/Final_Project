import math
import matplotlib
import numpy as np
import pandas as pd

import seaborn as sns
import time

from datetime import date
from matplotlib import pyplot as plt
from numpy.random import seed
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model
from tensorflow import train

#### Input params ##################

test_size = 0.2                # proportion of dataset to be used as test set
cv_size = 0.2                  # proportion of dataset to be used as cross-validation set

N_opt = 3                          # for feature at day t, we use lags from t-1, t-2, ..., t-N as features.
                               # initial value before tuning
lstm_units=50                  # lstm param. initial value before tuning.
dropout_prob=1                 # lstm param. initial value before tuning.
optimizer=train.AdamOptimizer(learning_rate = 0.003)
                               # lstm param. initial value before tuning.
epochs=50                       # lstm param. initial value before tuning.
batch_size=8                   # lstm param. initial value before tuning.

model_seed = 100

fontsize = 14
ticklabelsize = 14
####################################

# Set seeds to ensure same output results
seed(101)
set_random_seed(model_seed)

def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_x_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    """
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i-N:i])
        y.append(data[i])
    x = np.array(x)
    y = np.array(y)

    return x, y

def get_x_scaled_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    We scale x to have mean 0 and std dev 1, and return this.
    We do not scale y here.
    Inputs
        data     : pandas series to extract x and y
        N
        offset
    Outputs
        x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
        y        : target values. Not scaled
        mu_list  : list of the means. Same length as x_scaled and y
        std_list : list of the std devs. Same length as x_scaled and y
    """
    x_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(offset, len(data)):
        mu_list.append(np.mean(data[i-N:i]))
        std_list.append(np.std(data[i-N:i]))
        x_scaled.append((data[i-N:i]-mu_list[i-offset])/std_list[i-offset])
        y.append(data[i])
    x_scaled = np.array(x_scaled)
    y = np.array(y)

    return x_scaled, y, mu_list, std_list

def train_pred_eval_model(x_train_scaled, \
                          y_train_scaled, \
                          x_cv_scaled, \
                          y_cv, \
                          mu_cv_list, \
                          std_cv_list, \
                          lstm_units=50, \
                          dropout_prob=0.5, \
                          optimizer='adam', \
                          epochs=1, \
                          batch_size=1):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use LSTM here.
    Returns rmse, mape and predicted values
    Inputs
        x_train_scaled  : e.g. x_train_scaled.shape=(451, 9, 1). Here we are using the past 9 values to predict the next value
        y_train_scaled  : e.g. y_train_scaled.shape=(451, 1)
        x_cv_scaled     : use this to do predictions
        y_cv            : actual value of the predictions
        mu_cv_list      : list of the means. Same length as x_scaled and y
        std_cv_list     : list of the std devs. Same length as x_scaled and y
        lstm_units      : lstm param
        dropout_prob    : lstm param
        optimizer       : lstm param
        epochs          : lstm param
        batch_size      : lstm param
    Outputs
        rmse            : root mean square error
        mape            : mean absolute percentage error
        est             : predictions
    '''
    # Create the LSTM network
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1],1)))
    model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
    model.add(Dense(1))

    # Compile and fit the LSTM network
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    # Do prediction
    est_scaled = model.predict(x_cv_scaled)
    est = (est_scaled * np.array(std_cv_list).reshape(-1,1)) + np.array(mu_cv_list).reshape(-1,1)

    # Calculate RMSE and MAPE
#     print("x_cv_scaled = " + str(x_cv_scaled))
#     print("est_scaled = " + str(est_scaled))
#     print("est = " + str(est))
    rmse = math.sqrt(mean_squared_error(y_cv, est))
    mape = get_mape(y_cv, est)

    return rmse, mape, est
import os

directory_in_str = 'fang'

directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        print(filename)
        df = pd.read_csv(directory_in_str + '/' +filename, sep = ",")

        # Convert Date column to datetime
        df.loc[:, 'date'] = pd.to_datetime(df['date'],format='%m/%d/%Y')

        # Change all column headings to be lower case, and remove spacing
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

        # Get month of each sample
        df['month'] = df['date'].dt.month

        # Sort by datetime
        df.sort_values(by='date', inplace=True, ascending=True)

        # Get sizes of each of the datasets
        num_cv = int(cv_size*len(df))
        num_test = int(test_size*len(df))
        num_train = len(df) - num_cv - num_test
        # print("num_train = " + str(num_train))
        # print("num_cv = " + str(num_cv))
        # print("num_test = " + str(num_test))
        print(df.head())
        # Split into train, cv, and test
        train = df[:num_train][['date', 'adj_close']]
        cv = df[num_train:num_train+num_cv][['date', 'adj_close']]
        train_cv = df[:num_train+num_cv][['date', 'adj_close']]
        test = df[num_train+num_cv:][['date', 'adj_close']]

        # print("train.shape = " + str(train.shape))
        # print("cv.shape = " + str(cv.shape))
        # print("train_cv.shape = " + str(train_cv.shape))
        # print("test.shape = " + str(test.shape))

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(np.array(train['adj_close']).reshape(-1,1))
        # print("scaler.mean_ = " + str(scaler.mean_))
        # print("scaler.var_ = " + str(scaler.var_))

        # Split into x and y
        x_train_scaled, y_train_scaled = get_x_y(train_scaled, N_opt, N_opt)
        # print("x_train_scaled.shape = " + str(x_train_scaled.shape)) # (446, 7, 1)
        # print("y_train_scaled.shape = " + str(y_train_scaled.shape)) # (446, 1)


        # Scale the cv dataset
        # Split into x and y
        x_cv_scaled, y_cv, mu_cv_list, std_cv_list = get_x_scaled_y(np.array(train_cv['adj_close']).reshape(-1,1), N_opt, num_train)
        # print("x_cv_scaled.shape = " + str(x_cv_scaled.shape))
        # print("y_cv.shape = " + str(y_cv.shape))
        # print("len(mu_cv_list) = " + str(len(mu_cv_list)))
        # print("len(std_cv_list) = " + str(len(std_cv_list)))

        # Here we scale the train_cv set, for the final model
        scaler_final = StandardScaler()
        train_cv_scaled_final = scaler_final.fit_transform(np.array(train_cv['adj_close']).reshape(-1,1))
        # print("scaler_final.mean_ = " + str(scaler_final.mean_))
        # print("scaler_final.var_ = " + str(scaler_final.var_))

        # Create the LSTM network
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1],1)))
        model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
        model.add(LSTM(units=lstm_units))
        model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=2)

        # Split train_cv into x and y
        x_train_cv_scaled, y_train_cv_scaled = get_x_y(train_cv_scaled_final, N_opt, N_opt)

        # Split test into x and y
        x_test_scaled, y_test, mu_test_list, std_test_list = get_x_scaled_y(np.array(df['adj_close']).reshape(-1,1), N_opt, num_train+num_cv)
        rmse, mape, est = train_pred_eval_model(x_train_cv_scaled, \
                                              y_train_cv_scaled, \
                                              x_test_scaled, \
                                              y_test, \
                                              mu_test_list, \
                                              std_test_list, \
                                              lstm_units=lstm_units, \
                                              dropout_prob=dropout_prob, \
                                              optimizer=optimizer, \
                                              epochs=epochs, \
                                              batch_size=batch_size)

        est_df = pd.DataFrame({'date': df[num_train+num_cv:]['date'],
                               'est': est.reshape(-1)
                               })
        output = est_df.merge(test, on="date")
        output.to_csv('fang_lstm_out/'+filename)

        print("done")
