from typing import Tuple, Iterable

from collections import deque

import numpy as np
import pandas as pd

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

def make_multistep_sequences(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "close",
    n_steps: int = 50,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    #Add date as a column
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = df.index

    #Make sure there is enough rows to build sequences
    if len(df) < n_steps + k:
        raise ValueError(
            f"Not enough rows ({len(df)}) to build multistep sequences "
            f"with n_steps={n_steps} and k={k}. "
            f"Try reducing these values or use a longer dataset."
        )

    sequences = deque(maxlen=n_steps)

    X, y, indices = [], [], []

    target_values = df[target_column].values

    #Build sequences row by row
    for i, entry in enumerate(df[list(feature_columns)].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            #Ensure enough future steps remain
            if i + 1 + k <= len(target_values):
                #Collect the next k future values from target column
                future_targets = target_values[i + 1 : i + 1 + k]
                X.append(np.array(sequences))
                y.append(future_targets)
                indices.append(i)  #Store index for alignment

    #Get the last n_steps of features for future forecasting
    last_sequence = np.array(df[list(feature_columns)].tail(n_steps))

    return np.array(X), np.array(y), indices, last_sequence