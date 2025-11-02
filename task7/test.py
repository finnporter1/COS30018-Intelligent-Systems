from sentiment import (
    collect_news,
    analyse_sentiment,
    merge_with_stock_data,
    add_engineered_features,
    train_classifier,
    evaluate_model
)
from load_and_process_data import load_and_process_data

# Load stock data
data = load_and_process_data(
    ticker="QAN.AX",
    start_date="2023-01-01",
    end_date="2024-12-31",
    handle_nan="ffill_bfill",
    scale_features=False
)
price_df = data["df"].reset_index()

# Collect and score sentiment
news_df = collect_news('"Qantas Airways" OR "Qantas"', "2023-01-01", "2024-12-31")
daily_sent = analyse_sentiment(news_df)

# Merge and engineer features
merged_df = merge_with_stock_data(price_df, daily_sent)
merged_df = add_engineered_features(merged_df)

# Baseline model (no sentiment)
feature_cols_baseline = [
    "open",
    "return_1d", "sma_5", "ema_10", "volatility_5", "rsi_14"
]

model_base, results_base = train_classifier(merged_df, features=feature_cols_baseline)
print("\nBaseline model (no sentiment)")
evaluate_model(results_base)

# Model with sentiment scores
feature_cols_full = [
    "open",
    "return_1d", "sma_5", "ema_10", "volatility_5", "rsi_14",
    "sentiment_score", "sentiment_lag1", "sentiment_lag3", "sentiment_lag7"
]

model_full, results_full = train_classifier(merged_df, features=feature_cols_full)
print("\nModel with sentiment scores")
evaluate_model(results_full)