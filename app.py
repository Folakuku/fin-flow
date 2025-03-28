# Import necessary libraries
from typing_extensions import TypedDict
import pandas as pd
from pmdarima.arima import ARIMA
import requests
from textblob import TextBlob
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import plotly.graph_objs as go
from fastapi import FastAPI
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

# Define the state structure

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


class FinancialAnalysisState(TypedDict):
    symbol: str                  # Stock symbol (e.g., "AAPL")
    stock_prices: pd.DataFrame   # Historical stock prices
    news_articles: list[dict]    # List of news articles
    news_sentiment: float        # Average sentiment score of news
    model: ARIMA                 # Trained ARIMA model
    predictions: pd.Series       # Predicted prices for the next 7 days
    report_text: str             # Generated report text
    visualizations: list[dict]   # Visualization data (Plotly JSON)
    feedback: str                # Analyst's feedback

# Data Ingestion Node


def ingestion_node(state: FinancialAnalysisState) -> FinancialAnalysisState:
    symbol = state["symbol"]

    # Fetch stock prices from Alpha Vantage
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
    response = requests.get(url)
    data = response.json()

    # Parse stock prices into a DataFrame
    time_series = data.get("Time Series (Daily)", {})
    dates = []
    closes = []
    for date, values in time_series.items():
        dates.append(date)
        closes.append(float(values["4. close"]))
    stock_prices_df = pd.DataFrame(
        {"close": closes}, index=pd.to_datetime(dates))
    state["stock_prices"] = stock_prices_df.sort_index()

    # Fetch news articles from NewsAPI
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWSAPI_KEY}&language=en&sortBy=publishedAt&pageSize=20"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    state["news_articles"] = [
        {"title": article["title"], "content": article["content"]
            or "", "date": article["publishedAt"]}
        for article in articles
    ]
    return state

# Preprocessing Node


def preprocessing_node(state: FinancialAnalysisState) -> FinancialAnalysisState:
    # Process stock prices: ensure sorted index and handle missing values
    stock_prices = state["stock_prices"]
    stock_prices = stock_prices.asfreq(
        "D", method="ffill")  # Forward fill missing days

    # Calculate news sentiment
    articles = state["news_articles"]
    sentiments = [
        TextBlob(article["content"]).sentiment.polarity
        for article in articles if article["content"]
    ]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    state["news_sentiment"] = avg_sentiment
    state["stock_prices"] = stock_prices
    return state

# Market Analysis Node


def analysis_node(state: FinancialAnalysisState) -> FinancialAnalysisState:
    # Use last 5 years of data for training
    # Approx 5 years of trading days
    closing_prices = state["stock_prices"]["close"].iloc[-5*252:]
    from pmdarima import auto_arima
    model = auto_arima(closing_prices, seasonal=False, suppress_warnings=True)
    predictions = model.predict(n_periods=7)
    future_dates = pd.date_range(
        start=closing_prices.index[-1] + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )
    state["predictions"] = pd.Series(predictions, index=future_dates)
    state["model"] = model
    return state


# Report Generation Node
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-8b-8192",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = PromptTemplate(
    input_variables=["predictions", "sentiment"],
    template="Based on the predicted stock prices for the next week: {predictions}, and the recent news sentiment being {sentiment}, provide a concise analysis and advice for investors."
)


def report_generation_node(state: FinancialAnalysisState) -> FinancialAnalysisState:
    predictions_str = state["predictions"].round(2).to_string()
    sentiment_score = state["news_sentiment"]
    sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    report_text = llm.invoke(prompt.format(
        predictions=predictions_str, sentiment=sentiment))
    state["report_text"] = report_text

    # Create visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=state["stock_prices"].index[-30:],  # Last 30 days for context
        y=state["stock_prices"]["close"][-30:],
        name="Historical",
        mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=state["predictions"].index,
        y=state["predictions"],
        name="Predicted",
        mode="lines+markers",
        line=dict(dash="dash")
    ))
    fig.update_layout(
        title=f"Price Trend for {state['symbol']}", xaxis_title="Date", yaxis_title="Price")
    state["visualizations"] = [fig.to_json()]
    return state

# Review Node


def review_node(state: FinancialAnalysisState) -> FinancialAnalysisState:
    # This node runs after feedback is provided; for now, it just passes the state
    return state


# Set up FastAPI app
app = FastAPI()

# Set up LangGraph
graph = StateGraph(FinancialAnalysisState)
graph.add_node("ingestion", ingestion_node)
graph.add_node("preprocessing", preprocessing_node)
graph.add_node("analysis", analysis_node)
graph.add_node("report_generation", report_generation_node)
graph.add_node("review", review_node)

# Define edges
graph.add_edge("ingestion", "preprocessing")
graph.add_edge("preprocessing", "analysis")
graph.add_edge("analysis", "report_generation")
graph.add_edge("report_generation", "review")

# Set entry point
graph.set_entry_point("ingestion")

# Configure checkpointing for human-in-the-loop
checkpointer = MemorySaver()
compiled_graph = graph.compile(
    checkpointer=checkpointer, interrupt_before=["review"])

# FastAPI Endpoints


@app.post("/start_analysis/{symbol}")
async def start_analysis(symbol: str):
    """Start the financial analysis for a given stock symbol."""
    config = {"configurable": {"thread_id": symbol}}  # Unique thread ID
    initial_state = FinancialAnalysisState(symbol=symbol)
    for output in compiled_graph.stream(initial_state, config=config):
        if "review" in output:
            break
    current_state = checkpointer.get(config)
    return {
        "message": "Report generated, awaiting feedback",
        "report_text": current_state["report_text"],
        "visualizations": current_state["visualizations"]
    }


@app.post("/provide_feedback/{symbol}")
async def provide_feedback(symbol: str, feedback: str):
    """Submit feedback from an analyst to resume and complete the workflow."""
    config = {"configurable": {"thread_id": symbol}}
    current_state = checkpointer.get(config)
    if not current_state:
        return {"message": "No analysis found for this symbol"}
    current_state["feedback"] = feedback
    for output in compiled_graph.stream(current_state, config=config):
        pass  # Run to completion
    return {"message": "Feedback received and process completed"}

# Run the app with: uvicorn script_name:app --reload
