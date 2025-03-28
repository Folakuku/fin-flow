# ticker_api.py
from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route("/api/tickers", methods=["GET"])
def get_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        df = pd.read_html(str(table))[0]
        tickers = df["Symbol"].tolist()

        # Convert BRK.B → BRK-B for yfinance compatibility
        tickers = [t.replace('.', '-') for t in tickers]

        return jsonify(tickers)
    except Exception as e:
        print(f"[ERROR] Failed to fetch tickers: {e}")
        fallback = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA"]
        return jsonify(fallback), 200


if __name__ == "__main__":
    app.run(port=8000)
