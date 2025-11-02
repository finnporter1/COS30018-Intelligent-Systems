from sentiment import (
    collect_news,
    analyse_sentiment,
    merge_with_stock_data,
    add_engineered_features,
    train_classifier,
    evaluate_model
)
from load_and_process_data import load_and_process_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pytrends.request import TrendReq
import pandas as pd

# Load stock data
data = load_and_process_data(
    ticker="QAN.AX",
    start_date="2023-01-01",
    end_date="2024-12-31",
    handle_nan="ffill_bfill",
    scale_features=False
)
price_df = data["df"].reset_index()

# Collect and score sentiment with FinBERT
news_df = collect_news('"Qantas Airways" OR "Qantas"', "2023-01-01", "2024-12-31")
daily_sent = analyse_sentiment(news_df)

# FinBERT setup
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def finbert_sentiment(headline: str) -> float:
    inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).numpy()[0]
    # assume classes [negative, neutral, positive]
    return float(scores[2] - scores[0])

news_df["finbert_score"] = news_df["headline"].astype(str).apply(finbert_sentiment)
daily_finbert = news_df.groupby("date")["finbert_score"].mean().reset_index()
daily_finbert.rename(columns={"finbert_score": "finbert_sentiment"}, inplace=True)

# Merge both sentiment types
sentiment_merged = pd.merge(daily_sent, daily_finbert, on="date", how="outer").fillna(0)

# Google Trends feature
pytrends = TrendReq(hl='en-AU', tz=600)
pytrends.build_payload(["Qantas"], timeframe="2023-01-01 2024-12-31", geo="AU")
trend_df = pytrends.interest_over_time().reset_index()
trend_df = trend_df.rename(columns={"Qantas": "google_trend"})[["date", "google_trend"]]

# Merge price, sentiment and trends 
merged_df = merge_with_stock_data(price_df, sentiment_merged)
merged_df = pd.merge(merged_df, trend_df, left_on="Date", right_on="date", how="left")
merged_df["google_trend"] = merged_df["google_trend"].fillna(0)
merged_df = merged_df.drop(columns=["date"])

# Engineer features 
merged_df = add_engineered_features(merged_df)

# Train models (with sentiment and without sentiment)
# With sentiment + FinBERT + GoogleTrends
feature_cols_full = [
    "open", "return_1d", "sma_5", "ema_10", "volatility_5", "rsi_14",
    "sentiment_score", "sentiment_lag1", "sentiment_lag3", "sentiment_lag7",
    "finbert_sentiment", "google_trend"
]

model_full, results_full = train_classifier(merged_df, features=feature_cols_full)
print("\nExtension model (FinBERT + Google Trends)")
evaluate_model(results_full)

# Baseline without sentiment/trends
feature_cols_baseline = [
    "open", "return_1d", "sma_5", "ema_10", "volatility_5", "rsi_14"
]
model_base, results_base = train_classifier(merged_df, features=feature_cols_baseline)
print("\nBaseline model (no sentiment/trends)")
evaluate_model(results_base)