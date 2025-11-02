import os
import urllib.parse
import requests
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from gnews import GNews

# Data Collection
def collect_news(query: str, start: str, end: str,
                 out_path: str = "sentiment_data/news_data.csv") -> pd.DataFrame:

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Encode query to avoid URL control-character errors
    encoded_query = urllib.parse.quote_plus(query)

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    step = dt.timedelta(days=30)

    all_articles = []
    g = GNews(language='en', country='AU', max_results=100)

    while start_date < end_date:
        chunk_end = min(start_date + step, end_date)
        g.start_date, g.end_date = start_date, chunk_end
        print(f"Fetching '{query}' from {start_date} → {chunk_end}")
        try:
            chunk = g.get_news(encoded_query)
            all_articles.extend(chunk)
        except Exception as e:
            print(f"Chunk {start_date}→{chunk_end} failed: {e}")
        start_date = chunk_end

    if not all_articles:
        print("No articles found for the full period.")
        return pd.DataFrame(columns=["date", "headline", "source", "url"])

    df = pd.DataFrame([{
        "date": a.get("published date"),
        "headline": a.get("title"),
        "source": (
            a.get("publisher", {}).get("title")
            if isinstance(a.get("publisher"), dict)
            else a.get("publisher")
        ),
        "url": a.get("url")
    } for a in all_articles])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df.dropna(subset=["date"], inplace=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} total headlines → {out_path}")
    return df


# Sentiment Analysis
def analyse_sentiment(news_df: pd.DataFrame,
                      out_path: str = "sentiment_data/daily_sentiment.csv") -> pd.DataFrame:

    sia = SentimentIntensityAnalyzer()
    news_df["compound"] = news_df["headline"].astype(str).apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )
    daily = news_df.groupby("date")["compound"].mean().reset_index()
    daily.rename(columns={"compound": "sentiment_score"}, inplace=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily.to_csv(out_path, index=False)
    print(f"\nSaved daily sentiment scores → {out_path}")
    return daily


# Merge sentiment with stock price data
def merge_with_stock_data(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:

    df = price_df.copy()

    # Ensure 'Date' column exists
    if "Date" not in df.columns:
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    # Ensure sentiment 'date' column is datetime
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    # Merge sentiment onto stock data
    merged = pd.merge(df, sentiment_df, left_on="Date", right_on="date", how="left")
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0)
    merged.drop(columns=["date"], inplace=True)

    return merged

# Feature engineering

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Price based engineered features
    df["return_1d"] = df["close"].pct_change()  # Daily percentage change
    df["sma_5"] = df["close"].rolling(window=5).mean()  # 5-day Simple Moving Average
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()    # 10-day Exponential Moving Average
    df["volatility_5"] = df["close"].pct_change().rolling(window=5).std()   # 5-day rolling standard deviation
    df["rsi_14"] = compute_rsi(df["close"], window=14)  # 14-day Relative Strength Index

    # Sentiment based engineered features
    df["sentiment_lag1"] = df["sentiment_score"].shift(1)   # Previous day’s sentiment score
    df["sentiment_lag3"] = df["sentiment_score"].rolling(window=3).mean()   # 3-day average sentiment
    df["sentiment_lag7"] = df["sentiment_score"].rolling(window=7).mean()   # 7-day average sentiment

    # Drop any NaNs
    df.dropna(inplace=True)

    return df


# Model training and evaluation
def train_classifier(df: pd.DataFrame, features=None) -> tuple[RandomForestClassifier, dict]:

    if features is None:
        features = [
            "return_1d", "sma_5", "ema_10", "volatility_5", "rsi_14",
            "sentiment_score", "sentiment_lag1", "sentiment_lag3", "sentiment_lag7"
        ]

    # Create binary target
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    X = df[features]
    y = df["target"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return model, {"y_test": y_test, "y_pred": y_pred, "metrics": metrics}


def evaluate_model(results: dict, title="Confusion Matrix") -> None:
    
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    metrics = results["metrics"]

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()