import numpy as np
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from models import create_model
from sequences import make_multistep_sequences
from sklearn.metrics import mean_squared_error


def multistep_prediction(df_train, df_test, features, target, n_steps, k,
                         units=64, cell='LSTM', n_layers=2, dropout=0.2,
                         loss="mean_squared_error", optimizer="adam",
                         bidirectional=False, epochs=20, batch_size=32):

    # Create input sequences
    X_train, y_train, _, _ = make_multistep_sequences(df_train, features, target, n_steps, k)
    X_test, y_test, test_indices, _ = make_multistep_sequences(df_test, features, target, n_steps, k)

    # Dates for test predictions
    dates = df_test.iloc[test_indices]["date"].values

    # Build model
    cell_map = {"LSTM": LSTM, "GRU": GRU, "SimpleRNN": SimpleRNN}
    model = create_model(
        sequence_length=n_steps,
        n_features=X_train.shape[-1],
        units=units,
        cell=cell_map[cell],
        n_layers=n_layers,
        dropout=dropout,
        loss=loss,
        optimizer=optimizer,
        bidirectional=bidirectional,
        output_size=k,
    )

    # Train model
    hist = model.fit(X_train, y_train,
                     validation_data=(X_test, y_test),
                     epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "val_loss_last": hist.history["val_loss"][-1],
        "best_val_loss": min(hist.history["val_loss"]),
    }

    return y_test, y_pred, metrics, dates


def multivariate_prediction(df_train, df_test, features, target, n_steps, k,
                            n_features=None,
                            units=128, cell='GRU', n_layers=3, dropout=0.3,
                            loss="mean_squared_error", optimizer="adam",
                            bidirectional=False, epochs=20, batch_size=32):

    # Determine feature count
    n_features = n_features or len(features)
    
    # Filter out any unwanted columns ('future' column)
    filtered_train = df_train[list(features)].copy()
    filtered_test = df_test[list(features)].copy()

    # Create input sequences
    X_train, y_train, _, _ = make_multistep_sequences(filtered_train, features, target, n_steps, k)
    X_test, y_test, test_indices, _ = make_multistep_sequences(filtered_test, features, target, n_steps, k)

    # Dates for test predictions
    dates = df_test.iloc[test_indices]["date"].values

    # Build model
    cell_map = {"LSTM": LSTM, "GRU": GRU, "SimpleRNN": SimpleRNN}
    model = create_model(
        sequence_length=n_steps,
        n_features=n_features,
        units=units,
        cell=cell_map[cell],
        n_layers=n_layers,
        dropout=dropout,
        loss=loss,
        optimizer=optimizer,
        bidirectional=bidirectional,
        output_size=k,
    )

    # Train model
    hist = model.fit(X_train, y_train,
                     validation_data=(X_test, y_test),
                     epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "val_loss_last": hist.history["val_loss"][-1],
        "best_val_loss": min(hist.history["val_loss"]),
    }

    return y_test, y_pred, metrics, dates