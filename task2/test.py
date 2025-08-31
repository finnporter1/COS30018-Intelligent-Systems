#Test the load_and_process_data() function
from load_and_process_data import load_and_process_data

res = load_and_process_data(
    ticker="CBA.AX",
    start_date="2023-01-01",
    end_date="2023-12-31",
    feature_columns=("adjclose","volume","open","high","low"),
    target_column="adjclose",
    handle_nan="ffill_bfill",
    scale_features=True,
    split_method="date",
    test_size=0.2,
)

print("X_train:", res["X_train"].shape)
print("X_test:",  res["X_test"].shape)
print("y_train:", res["y_train"].shape)
print("y_test:",  res["y_test"].shape)
print("Scalers:", list(res["column_scaler"].keys()) if res["column_scaler"] else None)
print(res["df"].head())