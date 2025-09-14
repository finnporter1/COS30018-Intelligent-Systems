from load_and_process_data import load_and_process_data
from create_model import create_model, make_sequences
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
from sklearn.metrics import mean_squared_error
import pandas as pd, os
import numpy as np

#Model config parameters
FEATURES   = ("open","high","low","close","volume")
N_STEPS    = 50
LOOKUP     = 1
UNITS      = 256
N_LAYERS   = 2
CELL       = LSTM 
EPOCHS     = 20
DROPOUT    = 0.3
BATCH_SIZE = 64
BIDIRECTIONAL = False
MODEL_NAME = "task4_model_6"

#Load dataset
data = load_and_process_data(
    ticker="CBA.AX",
    start_date="2023-01-01", end_date="2024-12-31",
    feature_columns=FEATURES, target_column="close",
    handle_nan="ffill_bfill", scale_features=True,
    split_method="date", test_size=0.2,
)

#Output directory
output_dir = os.path.join("models", MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)

#Create DataFrames and sequences
df_train = pd.DataFrame(data["X_train"], columns=FEATURES)
df_train["future"] = pd.Series(data["y_train"]).shift(-LOOKUP)
df_test  = pd.DataFrame(data["X_test"],  columns=FEATURES)
df_test["future"]  = pd.Series(data["y_test"]).shift(-LOOKUP)

X_train, y_train, _ = make_sequences(df_train, FEATURES, N_STEPS, LOOKUP)
X_test,  y_test,  _ = make_sequences(df_test,  FEATURES, N_STEPS, LOOKUP)

#Build model
model = create_model(
    sequence_length=N_STEPS, n_features=X_train.shape[-1],
    units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT,
    loss="mean_absolute_error", optimizer="rmsprop", bidirectional=BIDIRECTIONAL
)

#Train model
hist = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
)

#Save model and metrics
model.save(os.path.join(output_dir, "model.keras"))

#Evaluate model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test).flatten()

#Additional metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
val_mae_last = hist.history["val_mean_absolute_error"][-1]
best_val_loss = min(hist.history["val_loss"])

#Print and save metrics
metrics = {
    "val_mae_last": val_mae_last,
    "eval_mae": float(mae),
    "eval_loss": float(loss),
    "best_val_loss": best_val_loss,
    "mse": mse,
    "rmse": rmse,
}

print(metrics)