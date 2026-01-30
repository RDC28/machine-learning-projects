import os
import json
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, render_template, request
from modules.stock_model import get_stock_insights, get_stock_data

app = Flask(__name__)

# Top 5 stocks – you can change this list anytime
TOP_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
MODEL_DIR = "models"  # same as models_dir in stock_model.py

# JSON paths
JSON_DIR = "json"
TOP_STOCKS_PATH = os.path.join(JSON_DIR, "top_stocks.json")
LOGS_PATH = os.path.join(JSON_DIR, "logs.json")
LOGS_STATS_PATH = os.path.join(JSON_DIR, "logs_data.json")


def get_today_ist() -> datetime:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist)


def get_today_ist_date_str() -> str:
    return get_today_ist().strftime("%Y-%m-%d")


def cleanup_models():
    """
    Delete model .pkl files for tickers we no longer care about:
    - Keep TOP_TICKERS
    - Keep tickers that still have 'pending' logs
    """
    if not os.path.exists(MODEL_DIR):
        return

    logs = load_logs()
    keep_tickers = set(t.upper() for t in TOP_TICKERS)

    # Keep any ticker that still has a pending prediction
    for entry in logs:
        if entry.get("status") == "pending":
            keep_tickers.add(entry["ticker"].upper())

    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith("_model.pkl"):
            continue

        ticker = fname.replace("_model.pkl", "").upper()
        full_path = os.path.join(MODEL_DIR, fname)

        if ticker not in keep_tickers:
            try:
                os.remove(full_path)
                print(f"Deleted old model: {full_path}")
            except Exception as e:
                print(f"Error deleting model {full_path}: {e}")

# ---------- Top stocks cache ----------

def load_cached_top_stocks():
    if not os.path.exists(TOP_STOCKS_PATH):
        return None

    try:
        with open(TOP_STOCKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("last_updated") == get_today_ist_date_str():
            return data.get("stocks", [])
    except Exception as e:
        print(f"Error reading cached JSON: {e}")

    return None


def generate_and_cache_top_stocks():
    stock_cards = []

    for ticker in TOP_TICKERS:
        try:
            info = get_stock_insights(ticker)

            change_pct = info.get("expected_change_pct", 0.0) or 0.0
            direction = "flat"
            if change_pct > 0:
                direction = "up"
            elif change_pct < 0:
                direction = "down"

            stock_cards.append({
                "ticker": info["ticker"],
                "last_close": round(info["last_close"], 2),
                "predicted_next_close": round(info["predicted_next_close"], 2),
                "change_pct": round(change_pct, 2),
                "change_abs": round(info["expected_change_abs"], 2),
                "direction": direction,
                "trend_label": info.get("trend_label", "neutral"),
            })
        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    stock_cards.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    os.makedirs(JSON_DIR, exist_ok=True)
    payload = {"last_updated": get_today_ist_date_str(), "stocks": stock_cards}

    try:
        with open(TOP_STOCKS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Error writing cached JSON: {e}")

    return stock_cards


# ---------- Logs helpers ----------

def load_logs():
    if not os.path.exists(LOGS_PATH):
        return []
    try:
        with open(LOGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("logs", [])
    except Exception as e:
        print(f"Error reading logs: {e}")
        return []


def save_logs(logs):
    os.makedirs(JSON_DIR, exist_ok=True)
    payload = {"logs": logs}
    try:
        with open(LOGS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Error saving logs: {e}")


def recompute_logs_stats(logs):
    evaluated = [e for e in logs if e.get("status") == "evaluated"]
    total_logged = len(logs)
    total_eval = len(evaluated)

    if total_eval == 0:
        stats = {
            "last_updated": get_today_ist_date_str(),
            "total_logged": total_logged,
            "total_evaluated": 0,
            "avg_abs_error": None,
            "avg_pct_error": None,
            "direction_accuracy": None,
        }
    else:
        avg_abs_error = sum(e["abs_error"] for e in evaluated) / total_eval
        avg_pct_error = sum(e["pct_error"] for e in evaluated) / total_eval
        dir_acc = (
            sum(1 for e in evaluated if e.get("direction_correct")) / total_eval * 100.0
        )
        stats = {
            "last_updated": get_today_ist_date_str(),
            "total_logged": total_logged,
            "total_evaluated": total_eval,
            "avg_abs_error": avg_abs_error,
            "avg_pct_error": avg_pct_error,
            "direction_accuracy": dir_acc,
        }

    with open(LOGS_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def load_logs_stats():
    if not os.path.exists(LOGS_STATS_PATH):
        logs = load_logs()
        return recompute_logs_stats(logs)
    try:
        with open(LOGS_STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading logs stats: {e}")
        logs = load_logs()
        return recompute_logs_stats(logs)


def add_log_entry(ticker, period, insights):
    logs = load_logs()
    new_id = max((e.get("id", 0) for e in logs), default=0) + 1

    today = get_today_ist().date()
    predicted_for = today + timedelta(days=1)
    predicted_for_str = predicted_for.strftime("%Y-%m-%d")

    entry = {
        "id": new_id,
        "ticker": ticker.upper(),
        "period": period,
        "created_at_iso": get_today_ist().isoformat(),
        "predicted_for_date": predicted_for_str,
        "last_close": float(insights["last_close"]),
        "predicted_close": float(insights["predicted_next_close"]),
        "status": "pending",
        "actual_close": None,
        "abs_error": None,
        "pct_error": None,
        "direction_correct": None,
    }

    logs.append(entry)
    # keep only last 100
    logs = logs[-100:]
    save_logs(logs)
    recompute_logs_stats(logs)
    return entry


def evaluate_pending_logs():
    logs = load_logs()
    today = get_today_ist().date()
    changed = False

    for entry in logs:
        if entry.get("status") != "pending":
            continue

        pred_date = datetime.strptime(entry["predicted_for_date"], "%Y-%m-%d").date()
        if today <= pred_date:
            continue  # too early to evaluate

        # fetch recent data and find the first close on/after pred_date
        try:
            df = get_stock_data(entry["ticker"], period="7d", interval="1d")
            df_dates = df["Date"].dt.date
            mask = df_dates >= pred_date
            df_sel = df[mask]
            if df_sel.empty:
                continue
            actual_close = float(df_sel.iloc[0]["Close"])
        except Exception as e:
            print(f"Error fetching actual close for log {entry['id']}: {e}")
            continue

        entry["actual_close"] = actual_close
        abs_error = abs(actual_close - entry["predicted_close"])
        entry["abs_error"] = abs_error
        entry["pct_error"] = abs_error / actual_close * 100.0 if actual_close != 0 else None

        # direction correctness
        pred_dir = 0
        actual_dir = 0
        if entry["predicted_close"] > entry["last_close"]:
            pred_dir = 1
        elif entry["predicted_close"] < entry["last_close"]:
            pred_dir = -1

        if actual_close > entry["last_close"]:
            actual_dir = 1
        elif actual_close < entry["last_close"]:
            actual_dir = -1

        entry["direction_correct"] = (pred_dir == actual_dir)
        entry["status"] = "evaluated"
        changed = True

    if changed:
        save_logs(logs)
        recompute_logs_stats(logs)

    # After updating logs & stats, clean up unused models
    cleanup_models()

    return logs



# ---------- Routes ----------

@app.route("/")
def home():
    stock_cards = load_cached_top_stocks()
    if not stock_cards:
        stock_cards = generate_and_cache_top_stocks()
    return render_template("index.html", stock_cards=stock_cards)


@app.route("/api/insights/<ticker>")
def api_insights(ticker):
    try:
        data = get_stock_insights(ticker)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["GET", "POST"])
def predict():
    ticker = None
    selected_period = "6mo"
    insights = None
    error = None
    log_message = None

    if request.method == "POST":
        action = request.form.get("action", "run")
        ticker = (request.form.get("ticker") or "").strip().upper()
        selected_period = request.form.get("period") or "6mo"

        if not ticker:
            error = "Please enter a ticker symbol."
        else:
            try:
                insights = get_stock_insights(
                    ticker,
                    recent_period=selected_period
                )

                if action == "log" and insights:
                    entry = add_log_entry(ticker, selected_period, insights)
                    log_message = f"Prediction saved. Will be evaluated for {entry['predicted_for_date']}."
            except Exception as e:
                error = str(e)

    return render_template(
        "predict.html",
        ticker=ticker,
        selected_period=selected_period,
        insights=insights,
        error=error,
        log_message=log_message,
    )


@app.route("/logs")
def logs_page():
    logs = evaluate_pending_logs()
    stats = load_logs_stats()
    # Sort logs newest first by predicted_for_date
    logs_sorted = sorted(
        logs,
        key=lambda e: (e.get("predicted_for_date", ""), e.get("id", 0)),
        reverse=True,
    )
    return render_template("logs.html", logs=logs_sorted, stats=stats)


@app.context_processor
def inject_year():
    return {"current_year": datetime.now().year}


if __name__ == "__main__":
    app.run(debug=True)
