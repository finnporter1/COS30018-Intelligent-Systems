from typing import Tuple, Iterable

from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Bidirectional

#Model creation
def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
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