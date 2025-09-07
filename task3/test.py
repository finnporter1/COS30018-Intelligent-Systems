from load_and_process_data import load_and_process_data
from visualise_stock import plot_candlestick
from visualise_stock import plot_boxplot

#Load real stock data (CBA.AX on ASX) using the existing pipeline
res = load_and_process_data(
    ticker="CBA.AX",
    start_date="2024-01-01",
    end_date="2024-12-31",
    feature_columns=("open", "high", "low", "close", "volume"),
    target_column="close",
    handle_nan="ffill_bfill",
    scale_features=False,    #Keep raw prices for candlestick charts
    split_method="date",
    test_size=0.2,
)

#Extract the cleaned dataframe with datetime index
df = res["df"]

#Candlestick chart with n=1
plot_candlestick(
    df=df,
    n=1,
    style='yahoo',
    volume=True,
    title="CBA.AX — Daily Candles (n=1)"
)

#Candlestick chart with n=5
plot_candlestick(
    df=df,
    n=5,
    style='yahoo',
    volume=True,
    title="CBA.AX — 5-Day Grouped Candles (n=5)"
)

#Boxplot chart with n=10
plot_boxplot(
    df,
    n=10,
    title="Boxplot of CBA stock prices (n=10)",
)

#Boxplot chart with n=1
plot_boxplot(
    df,
    n=1,
    title="Boxplot of CBA stock prices (n=1)",
)