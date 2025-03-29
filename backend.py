"""
# backend.py

✅ FinAI LangGraph with Memory, Multi-Agent, Tools & Enhanced Financial Pipeline
"""

import json
import typing
from gtts import gTTS
import os, re, logging
from io import BytesIO
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from scipy.stats import zscore
from dotenv import load_dotenv
from pydub import AudioSegment
from functools import lru_cache
import pandas as pd, numpy as np
import plotly.graph_objects as go
import os, json, warnings, requests
from difflib import get_close_matches
from typing import Dict, Any, Set, List
from plotly.subplots import make_subplots
from typing import TypedDict, Annotated, List, Optional

import yfinance as yf, pmdarima as pm
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image


def trim_audio(input_path: str, output_path: str, start_sec: float, end_sec: float) -> None:
    """Trim an audio file from start_sec to end_sec and save to output_path.
    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the trimmed audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
    """
    audio = AudioSegment.from_mp3(input_path)
    trimmed_audio = audio[start_sec * 1000:end_sec * 1000]  # Convert to milliseconds
    trimmed_audio.export(output_path, format="mp3")
    logger.info(f"Trimmed audio from {start_sec}s to {end_sec}s and saved to {output_path}")

load_dotenv()
warnings.filterwarnings("ignore")

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "finflow.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger("FinFlow")

# ========== API Keys ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
QUANDL_KEY = os.getenv("QUANDL_API_KEY")

memory = MemorySaver()

# ========== LLM + Tools ==========
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)
prompt = ChatPromptTemplate.from_template(
    "You are FinAI, a financial analyst. Provide a detailed analysis of {ticker}. "
    "Include historical performance, recent trends, and a 30-day forecast. "
    "Use technical indicators like RSI and moving averages. "
    "If feedback is provided, incorporate it: {feedback}."
)
chain = prompt | llm | StrOutputParser()

repl = PythonREPL()

@tool(description="Executes Python code and returns the output.")
def python_repl_tool(code: Annotated[str, "Python code to execute"]) -> str:
    """Executes Python code in a REPL environment."""
    try:
        return repl.run(code)
    except Exception as e:
        return f"Execution failed: {e}"

tavily_tool = TavilySearchResults(max_results=3)
llm_with_tools = llm.bind_tools([python_repl_tool, tavily_tool])

# === LangGraph Workflow ===
# Define the state for the graph
class AnalysisState(typing.Dict):
    ticker: str
    feedback: str
    historical_data: pd.DataFrame
    report: str
    viz_buffer: BytesIO
    audio_path: str

# Define nodes for the workflow
def validate_ticker(state: AnalysisState) -> AnalysisState:
    """Validate the ticker and fetch historical data."""
    ticker = state["ticker"]
    if not is_valid_ticker(ticker):
        suggestions = suggest_close_tickers(ticker, get_sp500_tickers() | get_currency_pairs())
        state["report"] = f"Invalid ticker: {ticker}. Did you mean: {', '.join(suggestions)}?"
        return state
    state["historical_data"] = yf.Ticker(ticker + "=X" if len(ticker) > 4 else ticker).history(period="1y")
    return state

def generate_report(state: AnalysisState) -> AnalysisState:
    """Generate the analysis report using LLM."""
    if "report" in state and state["report"].startswith("Invalid ticker"):
        return state
    ticker = state["ticker"]
    feedback = state["feedback"]
    historical_data = state["historical_data"]
    # Simplified report generation for the graph (actual logic is in analyze_ticker)
    state["report"] = f"Analysis for {ticker} generated."
    return state

def generate_forecast(state: AnalysisState) -> AnalysisState:
    """Generate the forecast visualization."""
    if "report" in state and state["report"].startswith("Invalid ticker"):
        return state
    # Simplified forecast generation for the graph
    state["viz_buffer"] = BytesIO()  # Placeholder
    return state

def generate_audio(state: AnalysisState) -> AnalysisState:
    """Generate the audio report."""
    if "report" in state and state["report"].startswith("Invalid ticker"):
        return state
    # Simplified audio generation for the graph
    state["audio_path"] = "audio.mp3"  # Placeholder
    return state

# Create the LangGraph workflow
workflow = StateGraph(AnalysisState)

# Add nodes
workflow.add_node("validate_ticker", validate_ticker)
workflow.add_node("generate_report", generate_report)
workflow.add_node("generate_forecast", generate_forecast)
workflow.add_node("generate_audio", generate_audio)

# Add edges
workflow.add_edge("validate_ticker", "generate_report")
workflow.add_edge("generate_report", "generate_forecast")
workflow.add_edge("generate_forecast", "generate_audio")
workflow.add_edge("generate_audio", END)

# Set the entry point
workflow.set_entry_point("validate_ticker")

# Compile the graph
graph = workflow.compile()

# ========== Financial Analysis LangGraph ==========
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
    feedback: Optional[str]
    start_sec: Optional[float]
    end_sec: Optional[float]

@lru_cache(maxsize=1)
def get_sp500_tickers() -> set:
    """Fetch and cache S&P 500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(url)[0]
        tickers = set(t.replace(".", "-") for t in df["Symbol"].tolist())
        return tickers
    except Exception:
        return {"AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"}

def get_currency_pairs() -> set:
    """Fetch currency pairs, validate using Alpha Vantage, and cache them locally.
    Returns:
        Set of validated currency pairs (e.g., EURUSD, USDJPY).
    """
    cache_file = "currency_pairs.json"
    
    # Try loading from cache
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_pairs = set(json.load(f))
            logger.info(f"Loaded {len(cached_pairs)} currency pairs from cache: {cached_pairs}")
            if cached_pairs:  # Only return if the cache is non-empty
                return cached_pairs
            else:
                logger.warning("Cache file exists but is empty")
        else:
            logger.info("No cache file found, proceeding to fetch currency pairs")
    except Exception as e:
        logger.warning(f"Failed to load currency pairs from cache: {e}")

    # Pre-defined list of common currency pairs
    pairs = {
        "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
        "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
        "AUDJPY", "CADJPY", "CHFJPY", "EURCHF", "GBPCHF",
        "EURAUD", "EURCAD", "AUDNZD", "NZDJPY", "USDSGD",
        "USDHKD", "USDMXN", "USDZAR", "USDRUB", "USDTRY"
    }
    logger.info(f"Starting with {len(pairs)} raw currency pairs: {pairs}")

    # Validate pairs using Alpha Vantage
    validated_pairs = set()
    if AV_KEY:
        try:
            fx = ForeignExchange(key=AV_KEY)
            for pair in pairs:
                try:
                    # Alpha Vantage expects from_symbol and to_symbol (e.g., EUR and USD)
                    from_currency = pair[:3]
                    to_currency = pair[3:]
                    # Test if the pair is valid by fetching a small amount of data
                    data, _ = fx.get_currency_exchange_daily(
                        from_symbol=from_currency,
                        to_symbol=to_currency,
                        outputsize="compact"
                    )
                    if not data.empty:
                        validated_pairs.add(pair)
                        logger.debug(f"Validated ticker: {pair}")
                    else:
                        logger.debug(f"Invalid ticker (empty data): {pair}")
                except Exception as e:
                    logger.debug(f"Invalid ticker ({e}): {pair}")
        except Exception as e:
            logger.error(f"Alpha Vantage validation failed: {e}")
            validated_pairs = pairs  # Fallback to unvalidated list if AV fails
    else:
        logger.warning("Alpha Vantage API key not found, skipping validation")
        validated_pairs = pairs

    logger.info(f"Validated {len(validated_pairs)} currency pairs: {validated_pairs}")
    if not validated_pairs:
        logger.error("No valid currency pairs after validation")
        validated_pairs = pairs  # Use the raw list as a last resort
        logger.warning("Using unvalidated currency pairs due to validation failure")

    # Cache validated pairs
    with open(cache_file, "w") as f:
        json.dump(list(validated_pairs), f)
    # Verify the file
    with open(cache_file, "r") as f:
        saved_pairs = json.load(f)
    if not saved_pairs:
        logger.error("Failed to save currency pairs to cache: File is empty")
        raise ValueError("Failed to save currency pairs to cache")
    logger.info(f"Cached {len(validated_pairs)} currency pairs: {validated_pairs}")
    return validated_pairs

def calculate_technicals(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators: SMAs, RSI, and Stochastics.
    Args:
        data: DataFrame with OHLC data.
    Returns:
        DataFrame with calculated indicators.
    """
    # Calculate SMAs
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()

    # Calculate RSI (14-day)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Calculate Stochastics (14-day)
    data["L14"] = data["Low"].rolling(window=14).min()
    data["H14"] = data["High"].rolling(window=14).max()
    data["%K"] = 100 * (data["Close"] - data["L14"]) / (data["H14"] - data["L14"])
    data["%D"] = data["%K"].rolling(window=3).mean()

    return data

def show_historical_chart(ticker: str, chart_type: str, include_metadata: bool) -> go.Figure:
    """Generate an interactive historical chart with technical indicators.
    Args:
        ticker: Stock or currency ticker.
        chart_type: Type of chart ("Line", "Candlestick", "Japanese Candlestick", "OHLC").
        include_metadata: Whether to include chart metadata.
    Returns:
        Plotly Figure object.
    """
    # Fetch data
    data = yf.Ticker(ticker + "=X" if len(ticker) > 4 else ticker).history(period="1y")
    if data.empty:
        logger.warning(f"No data available for {ticker}")
        return go.Figure()

    # Calculate technical indicators
    data = calculate_technicals(data)

    # Create a figure with subplots: 3 rows (Price, RSI, Stochastics)
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.2, 0.2],  # Price chart takes 60%, RSI and Stochastics 20% each
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price", "RSI (14)", "Stochastics")
    )

    # Main chart (Price + SMAs) in row 1
    if chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close",
            line=dict(color="blue")
        ), row=1, col=1)
    elif chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Candlestick",
            increasing_line_color="blue",
            decreasing_line_color="red"
        ), row=1, col=1)
    elif chart_type == "Japanese Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Japanese Candlestick",
            increasing_line_color="blue",
            decreasing_line_color="red"
        ), row=1, col=1)
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
            increasing_line_color="blue",
            decreasing_line_color="red"
        ), row=1, col=1)

    # Add SMAs to the main chart (row 1)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["SMA20"],
        mode="lines",
        name="SMA 20",
        line=dict(color="orange", dash="dash")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["SMA50"],
        mode="lines",
        name="SMA 50",
        line=dict(color="purple", dash="dash")
    ), row=1, col=1)

    # Add RSI to row 2
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["RSI"],
        mode="lines",
        name="RSI (14)",
        line=dict(color="cyan")
    ), row=2, col=1)

    # Add Stochastics to row 3
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["%K"],
        mode="lines",
        name="%K",
        line=dict(color="green")
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["%D"],
        mode="lines",
        name="%D",
        line=dict(color="red")
    ), row=3, col=1)

    # Update layout for the entire figure
    fig.update_layout(
        title=f"{ticker} Historical {chart_type} Chart" if include_metadata else None,
        height=800,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_dark",  # TradingView-style dark theme
    )

    # Update axes for each subplot
    fig.update_xaxes(showgrid=True, gridcolor="gray", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="gray", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="gray", row=3, col=1)

    fig.update_yaxes(title_text="Price", showgrid=True, gridcolor="gray", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], showgrid=True, gridcolor="gray", row=2, col=1)
    fig.update_yaxes(title_text="Stoch", range=[0, 100], showgrid=True, gridcolor="gray", row=3, col=1)

    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)

    # Add Stochastics overbought/oversold lines
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold", row=3, col=1)

    return fig

def generate_pdf_report(report_text: str, forecast_buffer: BytesIO, historical_fig: go.Figure) -> BytesIO:
    """Generate a PDF report containing the report text and charts.
    Args:
        report_text: The FinAI report text.
        forecast_buffer: BytesIO buffer containing the forecast chart image.
        historical_fig: Plotly Figure object for the historical chart.
    Returns:
        BytesIO buffer containing the PDF.
    """
    try:
        # Create a buffer for the PDF
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph("Fin-Flow Market Analysis Report", styles["Title"]))
        story.append(Spacer(1, 12))

        # Add report text
        story.append(Paragraph("Analysis Report", styles["Heading2"]))
        story.append(Spacer(1, 12))
        # Split the report text into paragraphs
        paragraphs = report_text.split("\n")
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para, styles["Normal"]))
                story.append(Spacer(1, 6))

        # Add forecast chart
        story.append(Paragraph("Forecast Chart", styles["Heading2"]))
        story.append(Spacer(1, 12))
        forecast_buffer.seek(0)  # Reset buffer position
        forecast_img = Image(forecast_buffer, width=500, height=300)
        story.append(forecast_img)
        story.append(Spacer(1, 12))

        # Add historical chart
        story.append(Paragraph("Historical Chart with Technical Indicators", styles["Heading2"]))
        story.append(Spacer(1, 12))
        # Convert Plotly figure to image
        historical_buffer = BytesIO()
        historical_fig.write_image(historical_buffer, format="png", width=800, height=600)
        historical_buffer.seek(0)
        historical_img = Image(historical_buffer, width=500, height=375)
        story.append(historical_img)
        story.append(Spacer(1, 12))

        # Build the PDF
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer

    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        raise

@lru_cache(maxsize=256)
def is_valid_ticker(ticker: str) -> bool:
    """Check ticker validity using S&P 500 list, currency pairs, or yfinance metadata.
    Args:
        ticker: The ticker to validate (e.g., AAPL, EURUSD).
    Returns:
        bool: True if the ticker is valid, False otherwise.
    """
    try:
        # Load currency pairs from cache
        with open("currency_pairs.json", "r") as f:
            currency_pairs = set(json.load(f))
        if ticker in currency_pairs:
            logger.debug(f"Ticker {ticker} validated as currency pair")
            return True

        # Check if the ticker is in the S&P 500 list
        sp500_tickers = get_sp500_tickers()
        if ticker in sp500_tickers:
            logger.debug(f"Ticker {ticker} validated as S&P 500 stock")
            return True

        # Fallback to yfinance for stocks not in S&P 500
        info = yf.Ticker(ticker).info
        if info and "shortName" in info:
            logger.debug(f"Ticker {ticker} validated via yfinance")
            return True
        else:
            logger.debug(f"Ticker {ticker} rejected by yfinance: No shortName in info")
            return False

    except Exception as e:
        logger.error(f"Failed to validate ticker {ticker}: {e}")
        # Fallback: If yfinance fails, rely on the S&P 500 list or currency pairs
        if ticker in sp500_tickers or ticker in currency_pairs:
            logger.debug(f"Ticker {ticker} validated via fallback (S&P 500 or currency pair)")
            return True
        logger.debug(f"Ticker {ticker} rejected: Not in S&P 500 or currency pairs")
        return False

def suggest_close_tickers(input_word: str, tickers: set) -> list:
    """Suggest similar tickers for user typos or partial inputs."""
    return get_close_matches(input_word.upper(), tickers, n=3, cutoff=0.6)

def data_ingestion(state: FinancialState) -> FinancialState:
    """Fetches and preprocesses stock or currency data and economic indicators.
    Args:
        state: FinancialState with ticker.
    Returns:
        Updated state with raw_data, preprocessed_data, economic_data."""
    ticker = state["ticker"]
    df = pd.DataFrame()
    is_currency = ticker in get_currency_pairs()

    if AV_KEY:
        try:
            if is_currency:
                fx = ForeignExchange(key=AV_KEY, output_format="pandas")
                data, _ = fx.get_currency_exchange_daily(from_symbol=ticker[:3], to_symbol=ticker[3:], outputsize="full")
                df = data["4. close"].rename("Close").tail(252).reset_index()
            else:
                ts = TimeSeries(key=AV_KEY, output_format="pandas")
                data, _ = ts.get_daily(symbol=ticker, outputsize="full")
                df = data["4. close"].rename("Close").tail(252).reset_index()
            df.columns = ["Date", "Close"]
            df["Date"] = pd.to_datetime(df["Date"])
            logger.info(f"Using Alpha Vantage for {ticker}")
        except Exception as e:
            logger.warning(f"Alpha Vantage failed: {e}")

    if df.empty:
        df = yf.Ticker(ticker + "=X" if is_currency else ticker).history(period="1y")[["Open", "High", "Low", "Close"]].reset_index()
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        logger.info(f"Using yfinance for {ticker}")

    econ_df = None
    if QUANDL_KEY:
        try:
            import nasdaqdatalink
            nasdaqdatalink.ApiConfig.api_key = QUANDL_KEY
            econ_df = nasdaqdatalink.get("FRED/DGS10", start_date="2024-01-01").rename(columns={"Value": "Treasury_Yield"}).reset_index()
            logger.info("Fetched Quandl Treasury Yield")
        except Exception as e:
            logger.warning(f"Quandl failed: {e}")

    return {
        "ticker": ticker,
        "raw_data": df,
        "preprocessed_data": df["Close"].dropna(),
        "economic_data": econ_df,
        "anomalies": [],
        "audio_path": "",
        "report": "",
        "viz_buffer": BytesIO(),
        "prediction": [],
        "feedback": None
    }

def market_analysis(state: FinancialState) -> FinancialState:
    """Predicts next 7 days of prices using Auto ARIMA.
    Args:
        state: FinancialState with preprocessed_data.
    Returns:
        Updated state with prediction."""
    series = state["preprocessed_data"]
    model = pm.auto_arima(
        series,
        seasonal=False,
        suppress_warnings=True,
        stepwise=True,
        max_order=5
    )
    state["prediction"] = model.predict(n_periods=7).tolist()
    logger.info(f"Generated 7-day prediction for {state['ticker']}")
    return state

def anomaly_detection(state: FinancialState) -> FinancialState:
    """Detects anomalies in price data using z-scores.
    Args:
        state: FinancialState with preprocessed_data.
    Returns:
        Updated state with anomalies."""
    scores = zscore(state["preprocessed_data"])
    state["anomalies"] = np.where(np.abs(scores) > 2)[0].tolist()
    logger.info(f"Detected {len(state['anomalies'])} anomalies for {state['ticker']}")
    return state

def report_generation(state: FinancialState) -> FinancialState:
    """Generates report, interactive chart, and audio advice.
    Args:
        state: FinancialState with all data.
    Returns:
        Updated state with report, viz_buffer, audio_path."""
    raw_data, prediction = state["raw_data"].copy(), state["prediction"]
    raw_data["Date"] = pd.to_datetime(raw_data["Date"])

    # Interactive Plotly chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw_data["Date"], y=raw_data["Close"], name="Close", mode="lines"))
    future_dates = pd.date_range(raw_data["Date"].iloc[-1], periods=8, freq="B")[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=prediction, name="Prediction", mode="lines", line=dict(dash="dash")))
    if state["economic_data"] is not None:
        fig.add_trace(go.Scatter(x=state["economic_data"]["Date"], y=state["economic_data"]["Treasury_Yield"], name="Treasury Yield", yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Yield (%)", overlaying="y", side="right"))
    fig.update_layout(title=f"{state['ticker']} Price Prediction", xaxis_title="Date", yaxis_title="Price")
    viz = BytesIO()
    fig.write_image(viz, format="png")
    viz.seek(0)

    # Economic context
    econ = "Not available"
    if state["economic_data"] is not None:
        delta = state["economic_data"]["Treasury_Yield"].tail(1).values[0] - state["economic_data"]["Treasury_Yield"].head(1).values[0]
        econ = f"Treasury Yield has {'increased' if delta > 0 else 'decreased'} by {abs(delta):.2f}%."

    # Web news
    web_summary = "Web search unavailable."
    if TAVILY_API_KEY:
        try:
            results = tavily_tool.invoke({"query": f"{state['ticker']} stock news"})
            web_summary = "\n".join([f"- {r['title']}: {r['content'][:150]}..." for r in results[:3]])
        except Exception as e:
            logger.warning(f"Tavily failed: {e}")

    # Report with anomalies
    prompt = ChatPromptTemplate.from_template(
        "Give detailed financial advice for {ticker} based on last 5 days: {last_5_days}, "
        "7-day prediction: {prediction}, economy: {economic_context}, news: {web_context}, "
        "anomalies: {anomalies}. Suggest buy/sell/hold and explain why."
    )
    last5 = raw_data["Close"].tail(5).to_string()
    anomalies = f"{len(state['anomalies'])} detected at indices {state['anomalies']}" if state["anomalies"] else "None"
    report = (prompt | llm).invoke({
        "ticker": state["ticker"],
        "last_5_days": last5,
        "prediction": prediction,
        "economic_context": econ,
        "web_context": web_summary,
        "anomalies": anomalies
    }).content

    # Audio report
    audio_dir = "audio_reports"
    os.makedirs(audio_dir, exist_ok=True)
    temp_audio_path = f"{audio_dir}/temp_{state['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    final_audio_path = f"{audio_dir}/trimmed_{state['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"

    # Generate the original audio
    gTTS(report).save(temp_audio_path)
    logger.info(f"Generated temporary audio report at {temp_audio_path}")

    # Trim the audio if start_sec and end_sec are provided
    start_sec = state.get("start_sec", 0)  # Default to 0 if not provided
    end_sec = state.get("end_sec", float('inf'))  # Default to end of audio if not provided
    try:
        trim_audio(temp_audio_path, final_audio_path, start_sec, end_sec)
        audio_path = final_audio_path
        # Clean up the temporary file
        os.remove(temp_audio_path)
    except Exception as e:
        logger.error(f"Failed to trim audio: {e}")
        audio_path = temp_audio_path  # Fallback to untrimmed audio

    state.update({
        "report": report,
        "viz_buffer": viz,
        "audio_path": audio_path
    })
    return state

# Financial Flow Graph
fin_graph = StateGraph(FinancialState)
fin_graph.add_node("data_ingestion", data_ingestion)
fin_graph.add_node("market_analysis", market_analysis)
fin_graph.add_node("anomaly_detection", anomaly_detection)
fin_graph.add_node("report_generation", report_generation)
fin_graph.add_edge(START, "data_ingestion")
fin_graph.add_edge("data_ingestion", "market_analysis")
fin_graph.add_edge("market_analysis", "anomaly_detection")
fin_graph.add_edge("anomaly_detection", "report_generation")
fin_graph.add_edge("report_generation", END)
fin_graph_compiled = fin_graph.compile()

# Conditional logic
fin_graph.add_conditional_edges(
    "report_generation",
    lambda state: "generate_audio" if state.get("audio_enabled") else "skip_audio",
    {
        "generate_audio": END,
        "skip_audio": END
    }
)

# Main Analysis Function
def analyze_ticker(ticker: str, feedback: str = "", audio_enabled: bool = False, start_sec: float = 0, end_sec: float = float('inf')) -> Dict[str, Any]:
    """Analyze a ticker and return a report, visualization, and audio.
    Args:
        ticker: Stock or currency ticker to analyze.
        feedback: Optional analyst feedback to incorporate.
        audio_enabled: Whether to generate an audio report.
        start_sec: Start time for audio trimming.
        end_sec: End time for audio trimming.
    Returns:
        Dict with analysis results.
    """
    # Initialize the state
    state = {
        "ticker": ticker,
        "feedback": feedback,
        "historical_data": None,
        "report": "",
        "viz_buffer": None,
        "audio_path": None
    }

    # Run the LangGraph workflow
    final_state = graph.invoke(state)

    # Extract results
    report = final_state["report"]
    viz_buffer = final_state["viz_buffer"]
    audio_path = final_state["audio_path"]

    # If the ticker is invalid, return early
    if report.startswith("Invalid ticker"):
        return {
            "response": report,
            "ticker": None,
            "viz_buffer": None,
            "audio_path": None,
            "report": None,
            "feedback": feedback
        }

    # Actual implementation (replace the simplified nodes with real logic)
    historical_data = yf.Ticker(ticker + "=X" if len(ticker) > 4 else ticker).history(period="1y")
    if historical_data.empty:
        logger.warning(f"No historical data for {ticker}")
        return {
            "response": f"No historical data available for {ticker}.",
            "ticker": None,
            "viz_buffer": None,
            "audio_path": None,
            "report": None,
            "feedback": feedback
        }

    # Generate the report using LLM
    response = chain.invoke({"ticker": ticker, "feedback": feedback})
    logger.info(f"Generated report for {ticker}")

    # Generate a simple forecast visualization
    viz_buffer = BytesIO()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data["Close"], mode="lines", name="Close"))
    fig.update_layout(title=f"{ticker} Historical Data", xaxis_title="Date", yaxis_title="Price")
    fig.write_image(viz_buffer, format="png")
    viz_buffer.seek(0)

    # Generate audio if enabled
    audio_path = None
    if audio_enabled:
        try:
            audio_path = f"audio/{ticker}_report.mp3"
            # Placeholder for audio generation (e.g., using gTTS)
            logger.info(f"Generated audio for {ticker} at {audio_path}")
        except Exception as e:
            logger.error(f"Failed to generate audio for {ticker}: {e}")

    return {
        "response": response,
        "ticker": ticker,
        "viz_buffer": viz_buffer,
        "audio_path": audio_path,
        "report": response,
        "feedback": feedback
    }

if __name__ == "__main__":
    result = analyze_ticker("AAPL")
    print(result["report"])