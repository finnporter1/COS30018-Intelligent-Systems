# load_and_process_data.py — Minimal C.2 implementation (no sequences)
# + automatic cleanup of any NaT/garbage row when reading cache

import os
from typing import Iterable, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_process_data(
    ticker: str,
    start_date: str,
    end_date: str,
    feature_columns: Iterable[str] = ("adjclose", "volume", "open", "high", "low"),
    target_column: str = "adjclose",
    handle_nan: str = "drop",             # "drop", "ffill", "bfill", "ffill_bfill"
    scale_features: bool = True,          # scale feature columns only
    split_method: str = "date",           # "date" or "random"
    test_size: float = 0.2,               # ratio for test split
    cache_dir: str = "data_cache",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """

    Returns:
      {
        "df": pd.DataFrame,              # cleaned dataframe (lowercase columns)
        "X_train": np.ndarray,           # features train
        "X_test": np.ndarray,            # features test
        "y_train": np.ndarray,           # target train
        "y_test": np.ndarray,            # target test
        "feature_columns": list[str],
        "target_column": str,
        "column_scaler": Dict[str, MinMaxScaler] | None  # scalers for features (if scaled)
      }
    """

    #Use a cached dataset or download a new one
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"{ticker.replace('.', '_')}_{start_date}_{end_date}.csv"
    )

    #If use_cache is set to True and the dataset exists, use it
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        # Set Date index and drop any accidental NaT row or fully-empty rows
        df = df.set_index("Date")
        df = df[~df.index.isna()]             # drop NaT index rows
        df = df.dropna(how="all")             # drop rows that are entirely NaN
    #Otherwise download the dataset
    else:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise ValueError(f"No data downloaded for {ticker} between {start_date} and {end_date}.")
        #Save with a Date column for future cached loads
        df.reset_index().to_csv(cache_path, index=False)

    #Normalise headers
    df.columns = [c.lower() for c in df.columns]

    #Ensure 'adjclose' exists (fallback to 'close')
    if "adjclose" not in df.columns:
        df["adjclose"] = df["close"]

    #Combine feature and target columns
    feature_columns = list(feature_columns)
    cols_all = list(feature_columns) + [target_column]

    #Use a unique list for NaN handling (avoids pandas duplicate-key error)
    cols_unique = list(dict.fromkeys(cols_all))

    #NaN handling
    if handle_nan == "drop":
        df = df.dropna(subset=cols_unique)
    elif handle_nan == "ffill":
        df[cols_unique] = df[cols_unique].ffill()
    elif handle_nan == "bfill":
        df[cols_unique] = df[cols_unique].bfill()
    elif handle_nan == "ffill_bfill":
        df[cols_unique] = df[cols_unique].ffill().bfill()
    else:
        raise ValueError("handle_nan must be one of {'drop','ffill','bfill','ffill_bfill'}")

    #Optional scaling
    if scale_features:
        column_scaler = {}
        for col in feature_columns:  #Scale feature columns only
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            column_scaler[col] = scaler  #Store column scaler for future use
    else:
        column_scaler = None

    #Assemble x, y
    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.float32)

    #Split methods
    if split_method == "date":
        #Preserves time order
        split_idx = int(len(df) * (1.0 - test_size))  #Calculate split index
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    elif split_method == "random":
        #Does not preserve time order
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42
        )
    else:
        raise ValueError("split_method must be 'date' or 'random'")

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "column_scaler": column_scaler,
    }
