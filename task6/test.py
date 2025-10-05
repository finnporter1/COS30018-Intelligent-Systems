from load_and_process_data import load_and_process_data
from models import create_dl_model, build_arima
from sequences import make_sequences
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ensemble import ensemble_average
import pandas as pd, os
import numpy as np

# Configuration
FEATURES   = ("open","high","low","close","volume")
TARGET     = "close"
N_STEPS    = 50
SEASONAL_PERIOD = 1  # weekly seasonality for SARIMA

ENSEMBLE_NAME = "model 7"

# Different DL model configs (all hyperparams included)
MODEL_CONFIGS = [
    {
        "CELL": LSTM, "UNITS": 128, "N_LAYERS": 2, "BIDIRECTIONAL": False,
        "EPOCHS": 15, "DROPOUT": 0.2, "BATCH_SIZE": 32, "NAME": "lstm"
    },
    {
        "CELL": GRU, "UNITS": 64, "N_LAYERS": 2, "BIDIRECTIONAL": False,
        "EPOCHS": 15, "DROPOUT": 0.2, "BATCH_SIZE": 32, "NAME": "gru"
    },
]

# Load data
data = load_and_process_data(
    ticker="CBA.AX",
    start_date="2023-01-01", end_date="2024-12-31",
    feature_columns=FEATURES, target_column=TARGET,
    handle_nan="ffill_bfill", scale_features=True,
    split_method="date", test_size=0.2,
)

df_train = pd.DataFrame(data["X_train"], columns=FEATURES)
df_train[TARGET] = data["y_train"]
df_train["future"] = pd.Series(data["y_train"]).shift(-1)   # add future column

df_test  = pd.DataFrame(data["X_test"], columns=FEATURES)
df_test[TARGET]  = data["y_test"]
df_test["future"]  = pd.Series(data["y_test"]).shift(-1)    # add future column

X_train, y_train, _ = make_sequences(df_train, FEATURES, N_STEPS)
X_test,  y_test,  _ = make_sequences(df_test,  FEATURES, N_STEPS)

# Train DL models
forecasts_dict = {}
metrics_summary = {}

for config in MODEL_CONFIGS:
    print(f"\nTraining model: {config['NAME']}")

    model = create_dl_model(
        sequence_length=N_STEPS, n_features=X_train.shape[-1],
        units=config["UNITS"], cell=config["CELL"], n_layers=config["N_LAYERS"],
        dropout=config["DROPOUT"], loss="mean_absolute_error", optimizer="rmsprop",
        bidirectional=config["BIDIRECTIONAL"], output_size=1   # single-step output
    )

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"],
        verbose=1
    )

    # Save model
    output_dir = os.path.join("models", config["NAME"])
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "model.keras"))

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test).flatten()

    # Store forecasts as Series with aligned index
    forecasts_dict[config["NAME"]] = pd.Series(y_pred, index=range(len(y_pred)))

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    val_mae_last = hist.history["val_mean_absolute_error"][-1]
    best_val_loss = min(hist.history["val_loss"])

    metrics = {
        "val_mae_last": float(val_mae_last),
        "eval_mae": float(mae),
        "eval_loss": float(loss),
        "best_val_loss": float(best_val_loss),
        "mse": float(mse),
        "rmse": float(rmse),
    }
    metrics_summary[config["NAME"]] = metrics

    print(f"Metrics for {config['NAME']}: {metrics}")

# Train SARIMA
print("\nTraining SARIMA")
train_series = pd.Series(data["y_train"])
forecast_periods = len(y_test)

sarima_forecast, sarima_model = build_arima(
    train_series, forecast_periods, seasonal_period=SEASONAL_PERIOD
)

# Store SARIMA forecast
forecasts_dict["sarima"] = sarima_forecast["ARIMA_Forecast"].reset_index(drop=True)

# Metrics for SARIMA
mse = mean_squared_error(y_test.flatten(), sarima_forecast["ARIMA_Forecast"])
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test.flatten(), sarima_forecast["ARIMA_Forecast"])

metrics_summary["sarima"] = {
    "mse": float(mse),
    "mae": float(mae),
    "rmse": float(rmse),
}
print("Metrics for SARIMA:", metrics_summary["sarima"])

# Ensemble creation
print("\nBuilding Ensemble")
final_forecast_df = ensemble_average(forecasts_dict)
ensemble_pred = final_forecast_df["Final_Forecast"].values

mse = mean_squared_error(y_test.flatten(), ensemble_pred)
mae = mean_absolute_error(y_test.flatten(), ensemble_pred)
rmse = np.sqrt(mse)

ensemble_metrics = {
    "mse": float(mse),
    "mae": float(mae),
    "rmse": float(rmse),
}
metrics_summary["ensemble"] = ensemble_metrics

print("Ensemble metrics:", ensemble_metrics)

# Save results
output_dir = os.path.join("ensembles", ENSEMBLE_NAME)
os.makedirs(output_dir, exist_ok=True)

results_df = pd.DataFrame(metrics_summary).T
results_path = os.path.join(output_dir, f"{ENSEMBLE_NAME}.csv")
results_df.to_csv(results_path)

print(f"\nSaved results to {results_path}")