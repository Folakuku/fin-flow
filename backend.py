# backend.py

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import yfinance as yf
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import nasdaqdatalink
from dotenv import load_dotenv
from io import BytesIO
import numpy as np
import warnings
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from datetime import datetime
from pathlib import Path
import json
import mplfinance as mpf
from scipy.stats import zscore
from tavily import TavilyClient

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")
load_dotenv()

# API key checks
required_keys = ["GROQ_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"Missing required API key: {key} in .env")

# Optional APIs
optional_keys = {
    "ALPHA_VANTAGE_API_KEY": "Alpha Vantage for financial data",
    "QUANDL_API_KEY": "Quandl for economic datasets",
    "TAVILY_API_KEY": "Tavily for web search",
}
for key, desc in optional_keys.items():
    if not os.getenv(key):
        print(f"Warning: {key} not set ({desc}). Falling back where applicable.")

tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

# 🎯 Get list of tickers
def get_sp500_tickers() -> List[str]:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        df = pd.read_html(str(table))[0]
        return [t.replace('.', '-') for t in df["Symbol"].tolist()]
    except Exception as e:
        print(f"[ERROR] Failed to fetch tickers: {e}")
        return ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA"]

# 🌐 LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=None
)

# 📊 State definition
class FinancialState(TypedDict):
    ticker: str
    raw_data: pd.DataFrame
    preprocessed_data: pd.Series
    economic_data: pd.DataFrame
    prediction: List[float]
    report: str
    viz_buffer: BytesIO
    anomalies: List[int]
    audio_path: str

# 📥 Node 1: Ingest
def data_ingestion(state: FinancialState) -> FinancialState:
    ticker = state["ticker"]
    df = pd.DataFrame()

    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_key:
        try:
            ts = TimeSeries(key=alpha_key, output_format="pandas")
            data, _ = ts.get_daily(symbol=ticker, outputsize="full")
            df = data["4. close"].rename("Close").tail(252).reset_index()
            df.rename(columns={"index": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception as e:
            print(f"[WARN] Alpha Vantage failed: {e}")

    if df.empty:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")[["Open", "High", "Low", "Close"]].reset_index()
        if "Date" not in df.columns:
            df = df.rename_axis("Date").reset_index()

    df.columns = [col.capitalize() if col.lower() == "date" else col for col in df.columns]

    econ_df = None
    if os.getenv("QUANDL_API_KEY"):
        try:
            nasdaqdatalink.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")
            econ_df = nasdaqdatalink.get("FRED/DGS10", start_date="2024-01-01").rename(
                columns={"Value": "Treasury_Yield"}).reset_index()
        except Exception as e:
            print(f"[WARN] Quandl failed: {e}")

    return {
        "ticker": ticker,
        "raw_data": df,
        "preprocessed_data": df["Close"].dropna(),
        "economic_data": econ_df,
        "anomalies": [],
        "audio_path": "",
        "report": "",
        "viz_buffer": BytesIO(),
        "prediction": []
    }

# 🔮 Node 2: Predict
def market_analysis(state: FinancialState) -> FinancialState:
    model = pm.auto_arima(state["preprocessed_data"], seasonal=False, stepwise=True)
    state["prediction"] = model.predict(n_periods=7).tolist()
    return state

# 🚨 Node 3: Anomaly Detection
def anomaly_detection(state: FinancialState) -> FinancialState:
    z_scores = zscore(state["preprocessed_data"])
    state["anomalies"] = np.where(np.abs(z_scores) > 2)[0].tolist()
    return state

# 📈 Node 4: Report + Viz + Audio
def report_generation(state: FinancialState) -> FinancialState:
    ticker, raw_data = state["ticker"], state["raw_data"].copy()
    chart_type, prediction = state.get("chart_type", "line").lower(), state["prediction"]
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])
    raw_data.set_index("Date", inplace=True)

    plot_data = raw_data.copy()
    plot_data.columns = [col.lower() for col in plot_data.columns]

    viz_buffer = BytesIO()
    ohlc_cols = {"open", "high", "low", "close"}

    if chart_type in {"candlestick", "japanese"} and ohlc_cols.issubset(set(plot_data.columns)):
        style = "charles" if chart_type == "candlestick" else "yahoo"
        mpf.plot(plot_data, type="candle", style=style, volume=False, savefig=viz_buffer)
    else:
        raw_data.reset_index(inplace=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(raw_data["Date"], raw_data["Close"], label="Historical Close")
        future_dates = pd.date_range(start=raw_data["Date"].iloc[-1], periods=8, freq="B")[1:]
        ax.plot(future_dates, prediction, label="Predicted Close", linestyle="--")
        ax.set_title(f"{ticker} Price Prediction")
        ax.legend()
        plt.savefig(viz_buffer, format="png")
        plt.close()

    viz_buffer.seek(0)

    econ_summary = "Not available"
    if state["economic_data"] is not None:
        econ = state["economic_data"]["Treasury_Yield"].tail(5)
        delta = econ.iloc[-1] - econ.iloc[0]
        trend = "increased" if delta > 0 else "decreased"
        econ_summary = f"10Y Treasury Yield has {trend} by {abs(delta):.2f}% in the last 5 days."

    web_summary = "Web results not available."
    if tavily_client:
        try:
            response = tavily_client.search(f"{ticker} stock news", search_depth="basic")
            if response["results"]:
                highlights = [
                    f"- {item['title']}: {item['content'][:150]}..."
                    for item in response["results"][:3]
                ]
                web_summary = "Recent headlines:\n" + "\n".join(highlights)
        except Exception as e:
            print(f"[WARN] Tavily search failed: {e}")

    prompt_text = Path("prompts/stock_advice.txt").read_text()
    prompt = ChatPromptTemplate.from_template(prompt_text)
    last5 = raw_data["Close"].tail(5).to_string()
    report = (prompt | llm).invoke({
        "ticker": ticker,
        "last_5_days": last5,
        "prediction": prediction,
        "economic_context": econ_summary,
        "web_context": web_summary
    }).content

    audio = gTTS(report)
    audio_path = f"audio_reports/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    os.makedirs("audio_reports", exist_ok=True)
    audio.save(audio_path)

    state.update({
        "report": report,
        "viz_buffer": viz_buffer,
        "audio_path": audio_path
    })

    return state

graph = StateGraph(FinancialState)
graph.add_node("data_ingestion", data_ingestion)
graph.add_node("market_analysis", market_analysis)
graph.add_node("anomaly_detection", anomaly_detection)
graph.add_node("report_generation", report_generation)
graph.add_edge(START, "data_ingestion")
graph.add_edge("data_ingestion", "market_analysis")
graph.add_edge("market_analysis", "anomaly_detection")
graph.add_edge("anomaly_detection", "report_generation")
graph.add_edge("report_generation", END)
app = graph.compile()

def run_analysis(ticker: str, chart_type: str = "line") -> FinancialState:
    state = {"ticker": ticker, "chart_type": chart_type}
    result = app.invoke(state)
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps({
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "summary": result["report"][:200]
        }) + "\n")
    return result
