import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import yfinance as yf
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries  # For Alpha Vantage
import nasdaqdatalink  # For Quandl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required API keys
required_keys = ["GROQ_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"Missing required API key: {key} in .env")

# Optional API keys with warnings
optional_keys = {
    "ALPHA_VANTAGE_API_KEY": "Alpha Vantage for financial data",
    "QUANDL_API_KEY": "Quandl for economic datasets",
    "TAVILY_API_KEY": "Tavily for web search",
}
for key, desc in optional_keys.items():
    if not os.getenv(key):
        print(f"Warning: {key} not set ({desc}). Falling back to yfinance where applicable.")

# Define the state to pass between nodes
class FinancialState(TypedDict):
    ticker: str
    raw_data: pd.DataFrame
    preprocessed_data: pd.DataFrame
    economic_data: pd.DataFrame
    prediction: List[float]
    report: str
    viz_path: str

# Initialize the LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Node 1: Data Ingestion and Preprocessing
def data_ingestion(state: FinancialState) -> FinancialState:
    """Fetch and preprocess financial data from yfinance, Alpha Vantage, and Quandl."""
    ticker = state["ticker"]
    df = pd.DataFrame()

    # Try Alpha Vantage first (more reliable for adjusted data)
    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_key:
        try:
            ts = TimeSeries(key=alpha_key, output_format="pandas")
            data, _ = ts.get_daily(symbol=ticker, outputsize="full")
            df = data["4. close"].rename("Close").tail(252).reset_index()  # ~1 year
            df["Date"] = pd.to_datetime(df["index"])
            df = df[["Date", "Close"]]
            print(f"Using Alpha Vantage for {ticker}")
        except Exception as e:
            print(f"Alpha Vantage failed: {e}. Falling back to yfinance.")
    
    # Fallback to yfinance if Alpha Vantage fails or isn’t set
    if df.empty:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        df = df[["Close"]].reset_index()
        print(f"Using yfinance for {ticker}")

    # Fetch economic data from Quandl (e.g., interest rates as context)
    quandl_key = os.getenv("QUANDL_API_KEY")
    economic_df = None
    if quandl_key:
        try:
            nasdaqdatalink.ApiConfig.api_key = quandl_key
            # Example: US 10-Year Treasury Yield as economic indicator
            economic_df = nasdaqdatalink.get("FRED/DGS10", start_date="2024-01-01")
            economic_df = economic_df.rename(columns={"Value": "Treasury_Yield"}).reset_index()
            print("Fetched Quandl economic data (Treasury Yield)")
        except Exception as e:
            print(f"Quandl failed: {e}. Proceeding without economic data.")

    return {
        "ticker": ticker,
        "raw_data": df,
        "preprocessed_data": df["Close"].dropna(),
        "economic_data": economic_df
    }

# Node 2: Market Analysis and Predictive Modeling
def market_analysis(state: FinancialState) -> FinancialState:
    """Predict next 7 days of prices using Auto ARIMA."""
    series = state["preprocessed_data"]
    model = pm.auto_arima(series, seasonal=False, stepwise=True, trace=False)
    forecast = model.predict(n_periods=7)
    return {
        "ticker": state["ticker"],
        "raw_data": state["raw_data"],
        "preprocessed_data": series,
        "economic_data": state["economic_data"],
        "prediction": forecast.tolist()
    }

# Node 3: Report Generation and Visualization
def report_generation(state: FinancialState) -> FinancialState:
    """Generate a report and visualization based on predictions and economic data."""
    ticker = state["ticker"]
    raw_data = state["raw_data"]
    prediction = state["prediction"]
    economic_data = state["economic_data"]

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(raw_data["Date"], raw_data["Close"], label="Historical Close")
    future_dates = pd.date_range(start=raw_data["Date"].iloc[-1], periods=8, freq="B")[1:]
    plt.plot(future_dates, prediction, label="Predicted Close", linestyle="--")
    if economic_data is not None:
        plt.twinx()
        plt.plot(economic_data["Date"], economic_data["Treasury_Yield"], color="green", label="10Y Treasury Yield", alpha=0.5)
        plt.ylabel("Yield (%)")
    plt.title(f"{ticker} Price Prediction (Next 7 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(loc="upper left")
    viz_path = f"{ticker}_prediction.png"
    plt.savefig(viz_path)
    plt.close()

    # Enhanced report with economic context
    prompt = ChatPromptTemplate.from_messages([(
        "user",
        "Given the following stock data for {ticker}:\n"
        "Last 5 days: {last_5_days}\n"
        "Predicted next 7 days: {prediction}\n"
        "Economic context (10Y Treasury Yield): {economic_context}\n"
        "Advise an analyst on potential actions based on this data."
    )])
    last_5_days = state["raw_data"]["Close"].tail(5).to_string()
    economic_context = economic_data["Treasury_Yield"].tail(5).to_string() if economic_data is not None else "Not available"
    chain = prompt | llm
    report = chain.invoke({
        "ticker": ticker,
        "last_5_days": last_5_days,
        "prediction": prediction,
        "economic_context": economic_context
    }).content

    return {
        "ticker": ticker,
        "raw_data": state["raw_data"],
        "preprocessed_data": state["preprocessed_data"],
        "economic_data": economic_data,
        "prediction": prediction,
        "report": report,
        "viz_path": viz_path
    }

# Build the LangGraph
graph = StateGraph(FinancialState)
graph.add_node("data_ingestion", data_ingestion)
graph.add_node("market_analysis", market_analysis)
graph.add_node("report_generation", report_generation)

# Define edges
graph.add_edge(START, "data_ingestion")
graph.add_edge("data_ingestion", "market_analysis")
graph.add_edge("market_analysis", "report_generation")
graph.add_edge("report_generation", END)

# Compile the graph
app = graph.compile()

# Run the system
if __name__ == "__main__":
    ticker = "AAPL"  # Example: Apple stock; change as needed
    initial_state = {"ticker": ticker}
    print(f"Analyzing {ticker}...")

    for state in app.stream(initial_state):
        if "report" in state:
            print("\n=== Final Report ===")
            print(state["report"])
            print(f"Visualization saved as: {state['viz_path']}")
        else:
            print(f"Step completed: {list(state.keys())[0]}")