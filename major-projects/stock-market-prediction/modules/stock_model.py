# stock_model.py

"""
Single module to:
- fetch stock data
- train a simple model
- save/load it
- predict the next-day closing price

Dependencies:
    pip install yfinance scikit-learn
"""

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ---------- DATA FETCHING ----------

def get_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker using yfinance.
    """
    ticker = ticker.upper().strip()

    df = yf.download(
    ticker,
    period=period,
    interval=interval,
    auto_adjust=True,  # or True if you prefer adjusted prices
    progress=False
)


    if df.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            "Check the symbol or try a different period."
        )

    df = df.reset_index()  # make Date a normal column
    return df


# ---------- HELPER: SUPERVISED DATA ----------

def _create_supervised_data(
    prices: np.ndarray,
    n_lags: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a 1D array of prices into X, y for supervised learning.
    X[t] = [price_{t-n_lags}, ..., price_{t-1}]
    y[t] = price_t
    """
    prices = np.ravel(prices).astype(float)  # force 1D

    X, y = [], []
    for i in range(n_lags, len(prices)):
        X.append(prices[i - n_lags:i])
        y.append(prices[i])

    X = np.array(X)
    y = np.array(y)

    # Just in case something made X 3D, flatten last dims
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    return X, y


# ---------- TRAINING + SAVING MODEL ----------

def train_and_save_model(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    n_lags: int = 5,
    models_dir: str = "models"
) -> str:
    """
    Train a simple LinearRegression model and save it as a pickle.

    Returns:
        str: Path to the saved pickle file.
    """
    df = get_stock_data(ticker, period=period, interval=interval)

    if "Close" not in df.columns:
        raise ValueError("Data does not contain 'Close' column.")

    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]

    prices = close_col.values.astype(float).ravel()

    if len(prices) <= n_lags + 1:
        raise ValueError("Not enough data to train the model.")

    X, y = _create_supervised_data(prices, n_lags=n_lags)
    print("X shape:", X.shape, "| y shape:", y.shape)

    # 90/10 train-test split
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"[{ticker.upper()}] MAE: {mae:.4f}, R²: {r2:.4f}")
    else:
        print(f"[{ticker.upper()}] Trained on all data, no test split.")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{ticker.upper()}_model.pkl")

    payload = {
        "model": model,
        "n_lags": n_lags,
        "ticker": ticker.upper()
    }

    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Model saved to: {model_path}")
    return model_path


# ---------- LOADING + PREDICTING ----------

def _load_model_payload(
    ticker: str,
    models_dir: str = "models"
) -> dict:
    """
    Load model payload for given ticker.
    """
    ticker = ticker.upper()
    model_path = os.path.join(models_dir, f"{ticker}_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found for '{ticker}'. Expected at {model_path}."
        )

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    return payload


def predict_next_close(
    ticker: str,
    auto_train: bool = True,
    train_period: str = "5y",
    recent_period: str = "1mo",
    interval: str = "1d",
    models_dir: str = "models"
) -> float:
    """
    Predict the next-day closing price for a ticker.

    If no model exists and auto_train=True, it will train one first.

    Args:
        ticker (str): e.g. "AAPL"
        auto_train (bool): Automatically train if model is missing.
        train_period (str): History used for training if needed.
        recent_period (str): Recent period used to build prediction window.
        interval (str): Data interval for both training + prediction.
        models_dir (str): Directory for models.

    Returns:
        float: Predicted next close.
    """
    ticker = ticker.upper()

    # 1. Ensure model exists
    try:
        payload = _load_model_payload(ticker, models_dir=models_dir)
    except FileNotFoundError:
        if not auto_train:
            raise
        print(f"No model for {ticker}, training one now...")
        train_and_save_model(
            ticker,
            period=train_period,
            interval=interval,
            n_lags=5,
            models_dir=models_dir
        )
        payload = _load_model_payload(ticker, models_dir=models_dir)

    model = payload["model"]
    n_lags = payload["n_lags"]

    # 2. Get recent data
    df_recent = get_stock_data(ticker, period=recent_period, interval=interval)

    if "Close" not in df_recent.columns:
        raise ValueError("Recent data does not contain 'Close' column.")

    close_col = df_recent["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]

    closes = close_col.values.astype(float)

    if len(closes) < n_lags:
        raise ValueError(
            f"Not enough recent data to predict. Need at least {n_lags} values."
        )

    last_window = closes[-n_lags:]
    X_input = last_window.reshape(1, -1)

    # 3. Predict
    next_price = model.predict(X_input)[0]
    return float(next_price)

def get_stock_insights(
    ticker: str,
    auto_train: bool = True,
    train_period: str = "5y",
    recent_period: str = "6mo",
    interval: str = "1d",
    models_dir: str = "models"
) -> dict:
    """
    Build a small 'insights' summary for a stock to help the user think
    about their decision.

    Returns a dict like:
        {
            "ticker": "AAPL",
            "last_close": 190.12,
            "predicted_next_close": 191.05,
            "expected_change_abs": 0.93,
            "expected_change_pct": 0.49,
            "ma_7": ...,
            "ma_30": ...,
            "ma_90": ...,
            "position_vs_30d_pct": ...,
            "daily_volatility_30d_pct": ...,
            "trend_label": "bullish" / "bearish" / "neutral"
        }
    """
    ticker = ticker.upper()

    # 1. Get recent data
    df_recent = get_stock_data(ticker, period=recent_period, interval=interval)

    if "Close" not in df_recent.columns:
        raise ValueError("Recent data does not contain 'Close' column.")

    close_col = df_recent["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]

    closes = close_col.astype(float)
    last_close = float(closes.iloc[-1])

    # 2. Get model prediction (this will auto-train if needed)
    predicted_next = predict_next_close(
        ticker=ticker,
        auto_train=auto_train,
        train_period=train_period,
        recent_period=recent_period,
        interval=interval,
        models_dir=models_dir
    )

    expected_change_abs = predicted_next - last_close
    expected_change_pct = (predicted_next / last_close - 1.0) * 100.0

    # 3. Moving averages (trend context)
    def safe_ma(series: pd.Series, window: int):
        if len(series) < window:
            return None
        return float(series.tail(window).mean())

    ma_7 = safe_ma(closes, 7)
    ma_30 = safe_ma(closes, 30)
    ma_90 = safe_ma(closes, 90)

    position_vs_30d_pct = None
    if ma_30 is not None and ma_30 != 0:
        position_vs_30d_pct = (last_close / ma_30 - 1.0) * 100.0

    # 4. Recent volatility (last 30 daily returns)
    daily_volatility_30d_pct = None
    if len(closes) > 2:
        returns = closes.pct_change().dropna()
        if len(returns) >= 5:  # at least a few points
            last_30 = returns.tail(30)
            daily_volatility_30d_pct = float(last_30.std() * 100.0)

    # 5. Simple trend label (very heuristic, just for UX)
    trend_score = 0

    # Short vs medium trend
    if ma_7 is not None and ma_30 is not None:
        if ma_7 > ma_30:
            trend_score += 1
        elif ma_7 < ma_30:
            trend_score -= 1

    # Current price vs short MA
    if ma_7 is not None:
        if last_close > ma_7:
            trend_score += 1
        elif last_close < ma_7:
            trend_score -= 1

    # Model's predicted move
    if expected_change_pct > 0:
        trend_score += 1
    elif expected_change_pct < 0:
        trend_score -= 1

    if trend_score >= 2:
        trend_label = "bullish"
    elif trend_score <= -2:
        trend_label = "bearish"
    else:
        trend_label = "neutral"

    insights = {
        "ticker": ticker,
        "last_close": last_close,
        "predicted_next_close": predicted_next,
        "expected_change_abs": expected_change_abs,
        "expected_change_pct": expected_change_pct,
        "ma_7": ma_7,
        "ma_30": ma_30,
        "ma_90": ma_90,
        "position_vs_30d_pct": position_vs_30d_pct,
        "daily_volatility_30d_pct": daily_volatility_30d_pct,
        "trend_label": trend_label,
    }

    return insights

# ---------- QUICK TEST ----------

if __name__ == "__main__":
    ticker = "AAPL"
    print(f"Training and generating insights for {ticker}...")
    info = get_stock_insights(ticker)
    from pprint import pprint
    pprint(info)
