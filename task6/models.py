from typing import Tuple, Iterable

from collections import deque

import numpy as np
import pandas as pd
import pmdarima as pm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Bidirectional

#Model creation
def create_dl_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False, output_size=1):
    model = Sequential()
    #Specify input once, remove batch_input_shape from the first RNN layer
    model.add(tf.keras.Input(shape=(sequence_length, n_features)))

    if n_layers == 1:
        #Single layer, no need for return_sequences
        if bidirectional:
            model.add(Bidirectional(cell(units, return_sequences=False)))
        else:
            model.add(cell(units, return_sequences=False))
    else:
        for i in range(n_layers):
            if i == 0:
                #First layer, return sequences to pass to next layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            elif i == n_layers - 1:
                #Last layer, no need to return sequences
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                #Hidden layers, must return sequences
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            #Add dropout after each layer
            model.add(Dropout(dropout))
    model.add(Dense(output_size, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

# Create ARIMA/SARIMA model
def build_arima(train_series, forecast_periods, seasonal_period=7):

    stepwise_fit = pm.auto_arima(
        train_series,
        start_p=1, start_q=1,
        max_p=3, max_d=2, max_q=3,
        m=seasonal_period,
        start_P=0, start_Q=0,
        max_P=3, max_D=3, max_Q=3,
        seasonal=True,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    forecast = pd.DataFrame(stepwise_fit.predict(n_periods=forecast_periods))
    forecast.columns = ['ARIMA_Forecast']
    
    return forecast, stepwise_fit