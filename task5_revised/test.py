from load_and_process_data import load_and_process_data
from predictions import multistep_prediction, multivariate_prediction
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Model configuration parameters
FEATURES = ("open", "high", "low", "close", "volume", "adjclose")
TARGET = "close"
N_STEPS = 50
LOOKUP = 1
K = 7  # Number of days to predict into the future

EPOCHS = 20
BATCH_SIZE = 32
MODEL_NAME = "task5_model_k=7"

# Load dataset
data = load_and_process_data(
    ticker="CBA.AX",
    start_date="2023-01-01", end_date="2024-12-31",
    feature_columns=FEATURES, target_column=TARGET,
    handle_nan="ffill_bfill", scale_features=True,
    split_method="date", test_size=0.2,
)

# Create DataFrames for train/test
df_train = pd.DataFrame(data["X_train"], columns=FEATURES)
df_train["future"] = pd.Series(data["y_train"]).shift(-LOOKUP)
df_test = pd.DataFrame(data["X_test"], columns=FEATURES)
df_test["future"] = pd.Series(data["y_test"]).shift(-LOOKUP)

df_train["date"] = data["train_dates"]
df_test["date"] = data["test_dates"]

# Run Univariate Multistep Prediction
print("\nRunning univariate multistep prediction...\n")
uni_features = ["close"]
y_test_uni, y_pred_uni, metrics_uni, uni_dates = multistep_prediction(
    df_train, df_test, uni_features, TARGET, N_STEPS, K,
    epochs=EPOCHS, batch_size=BATCH_SIZE
)

# Run Multivariate Multistep Prediction
print("\nRunning multivariate multistep prediction...\n")
y_test_multi, y_pred_multi, metrics_multi, multi_dates = multivariate_prediction(
    df_train, df_test, FEATURES, TARGET, N_STEPS, K,
    n_features=6,
    epochs=EPOCHS, batch_size=BATCH_SIZE
)

# Plot results

# Create output directory
os.makedirs("plots", exist_ok=True)

# Align all plots to full test date range
test_dates = df_test["date"].values
actual_close = df_test["close"].values

# Univariate plot
plt.figure(figsize=(12, 6))
plt.plot(test_dates[:len(y_test_uni)], y_test_uni[:, 0], label="Actual (1-step)")
plt.plot(test_dates[:len(y_pred_uni)], y_pred_uni[:, 0], label="Predicted (1-step)")
plt.title("Univariate Multistep Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("plots/univariate_prediction.png", dpi=300)
plt.close()

# Multivariate plot
plt.figure(figsize=(12, 6))
plt.plot(test_dates[:len(y_test_multi)], y_test_multi[:, 0], label="Actual (1-step)")
plt.plot(test_dates[:len(y_pred_multi)], y_pred_multi[:, 0], label="Predicted (1-step)")
plt.title("Multivariate Multistep Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("plots/multivariate_prediction.png", dpi=300)
plt.close()

# Multi-step segmented forecast (restricted to same range as 1-step)
plt.figure(figsize=(12, 6))

actual_subset = y_test_uni[:, 0]
actual_dates_subset = test_dates[:len(y_test_uni)]

plt.plot(actual_dates_subset, actual_subset, label="Actual (1-step range)", color="blue", linewidth=1.5)

# Define the last valid date based on the single-step range
max_plot_date = actual_dates_subset[-1]

# Plot multi-step prediction segments that fit within the same range
for i in range(len(y_pred_multi)):
    start_idx = i + 1
    end_idx = i + 1 + K
    if end_idx <= len(test_dates):
        future_dates = test_dates[start_idx:end_idx]

        # Only plot segments within the single-step time window
        if future_dates[0] <= max_plot_date:
            plt.plot(future_dates, y_pred_multi[i], color="orange", alpha=0.4)

plt.title(f"Multivariate Multi-step Forecasts ({K}-day windows)")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend(["Actual (1-step range)", "Predicted Segments"], loc="best")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

# Force x-axis to match the single-step range
plt.xlim(actual_dates_subset[0], max_plot_date)

plt.tight_layout()
plt.savefig("plots/multivariate_segments.png", dpi=300)
plt.close()

print("All plots saved to: plots/")

# Summary
print("\nUnivariate metrics:", metrics_uni)
print("Multivariate metrics:", metrics_multi)