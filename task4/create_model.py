from typing import Tuple, Iterable

from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Bidirectional

#Sequence creation
def make_sequences(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    n_steps: int = 50,
    lookup_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    #Add date as a column
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = df.index

    #Last `lookup_step` columns contains NaN in future column
    #Get them before droping NaNs
    last_sequence = np.array(df[list(feature_columns)].tail(lookup_step))

    #Drop NaNs ONLY from columns that actually exist
    subset_cols = [c for c in (list(feature_columns) + ['future']) if c in df.columns]
    if not subset_cols:
        raise ValueError(
            "None of the expected columns are present when dropping NaNs. "
            f"df.columns={list(df.columns)}"
        )
    df = df.dropna(subset=subset_cols)

    #Make sure we have enough rows to form at least one sequence
    if len(df) < n_steps + 1:
        raise ValueError(
            f"Not enough rows ({len(df)}) to build sequences with n_steps={n_steps} "
            f"and lookup_step={lookup_step}. Try reducing N_STEPS or LOOKUP_STEP, "
            f"or use a longer date range."
        )

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[list(feature_columns) + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    #Get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)

    #Construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    #Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y, last_sequence


#Model creation
def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
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
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model