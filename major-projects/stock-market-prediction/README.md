# rcStocks — Stock Market Prediction Web App

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blueviolet)](https://rcstocks.onrender.com)

A Flask web application that predicts next-day stock prices using machine learning. Features a clean dark UI, IST timezone support, daily caching, and a complete accuracy tracking system.

---

## Features

### Next-Day Price Predictions

The app pulls historical OHLC data from Yahoo Finance and trains a Linear Regression model for each ticker. Models are saved locally for reuse, so predictions are fast after the first run.

Each prediction includes:
- Last closing price
- Predicted next close
- Expected move (absolute value and percentage)
- Direction indicator (Up / Down / Flat)
- Short-term trend analysis using 7, 30, and 90-day moving averages

### Homepage Dashboard

The landing page displays predictions for 5 major tech stocks (AAPL, MSFT, AMZN, GOOGL, TSLA). These are cached once per day in IST timezone to avoid hammering the API, and the results are stored in `json/top_stocks.json`.

### Prediction Logging & Accuracy Tracking

After making a prediction, you can log it to track how accurate the model is over time. Here's how it works:

- **Add predictions** — Save any prediction to the log database
- **Auto-evaluation** — When you visit `/logs`, yesterday's predictions are automatically checked against actual closing prices
- **Error metrics** — Calculates absolute error, percentage error, and direction accuracy
- **History limit** — Keeps the latest 100 entries in `json/logs.json`

The logs page also shows aggregated statistics:
- Average absolute error
- Average percent error  
- Direction prediction accuracy
- Best and worst predictions
- Full log history

All stats are computed and cached in `json/logs_data.json`.

### Automatic Model Cleanup

The app automatically deletes unused model files to keep things tidy. It only keeps models for:
- The 5 homepage tickers
- Any tickers with pending predictions in the logs

Everything else gets cleaned up automatically.

### UI/UX

Built with Bootstrap 5 and a custom dark theme. The color scheme uses `#181818` for the background with gold accents (`#dba400`). Every page includes an IST live clock and follows a consistent card-based layout via `layout.html`.

---

## Tech Stack

**Backend**
- Flask (Python)
- Scikit-learn (Linear Regression)
- yfinance for market data
- JSON-based caching system

**Frontend**
- Bootstrap 5
- Jinja2 templating
- Custom CSS with dark theme

**Storage**
- `models/` — trained ML models (.pkl files)
- `json/top_stocks.json` — daily cached predictions
- `json/logs.json` — prediction history
- `json/logs_data.json` — accuracy statistics

---

## Project Structure

```
project/
│
├── app.py                    # Flask routes, caching, and log management
├── modules/
│   └── stock_model.py        # Data fetching, training, and prediction logic
│
├── templates/
│   ├── layout.html           # Base template with navbar and footer
│   ├── index.html            # Homepage with top 5 predictions
│   ├── predict.html          # Individual stock prediction page
│   └── logs.html             # Statistics and log viewer
│
├── models/                   # Trained model files (auto-managed)
│
└── json/
    ├── top_stocks.json       # Cached daily predictions
    ├── logs.json             # Prediction logs
    └── logs_data.json        # Aggregated metrics
```

---

## Installation

Clone the repository:
```bash
git clone https://github.com/RDC28/your-repo.git
cd your-repo
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
python app.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## Screenshots

**Homepage**  
![Homepage](img/home.png)

**Prediction Page**  
![Predict Page](img/prediction.png)

**Logs & Statistics**  
![Logs Page](img/logs.png)

---

## How the Model Works

**Algorithm:** Linear Regression (simple baseline model)

**Input features:** The last 5 closing prices

**Output:** Next-day predicted closing price, along with:
- Expected move and percentage change
- Predicted direction
- Moving average signals and volatility indicators

**Evaluation metrics** (tracked per log entry):
- Absolute error
- Percentage error  
- Direction accuracy

---

## Disclaimer

This is an educational project built to explore machine learning applications in finance. The predictions are based purely on historical price patterns and **should not be used for actual trading or investment decisions**.

---

## Contact

**Author:** @RDC28  
**GitHub:** [https://github.com/RDC28](https://github.com/RDC28)  
**LinkedIn:** [rchavda28](https://linkedin.com/rchavda28)
**Email:** [rchavda2005@outlook.com](mailto:rchavda2005@outlook.com)
