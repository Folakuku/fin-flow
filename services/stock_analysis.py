import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
import google.generativeai as genai
from .web_scraper import WebScraper
from .models import FinancialAnalysisState


class StockAnalysisService:
    def __init__(self, gemini_api_key: str):
        """
        Initialize the Stock Analysis Service

        :param gemini_api_key: API key for Google Generative AI
        """
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        self.web_scraper = WebScraper()

    def fetch_and_preprocess_data(self, stock_symbol: str) -> FinancialAnalysisState:
        """
        Fetch stock data and preprocess it

        :param stock_symbol: Stock symbol to analyze
        :return: FinancialAnalysisState object
        """
        try:
            # Fetch stock data from Yahoo Finance
            stock_data = yf.download(stock_symbol, period='1mo', interval='1d')

            # Preprocess data
            preprocessed_data = self._preprocess_data(stock_data)

            # Scrape news
            news = self.web_scraper.scrape_financial_news(stock_symbol)

            return FinancialAnalysisState(
                stock_symbol=stock_symbol,
                raw_data=stock_data.to_dict(),
                preprocessed_data=preprocessed_data.to_dict(),
                news=news
            )
        except Exception as e:
            return FinancialAnalysisState(
                stock_symbol=stock_symbol,
                error=str(e)
            )

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess stock market data

        :param data: Raw stock market data
        :return: Preprocessed DataFrame
        """
        if data is None or data.empty:
            return pd.DataFrame()

        data = data.dropna()
        data.index = pd.to_datetime(data.index)  # Ensure datetime index
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(1 + data['Returns'])
        return data

    def generate_predictions(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Generate stock price predictions using ARIMA

        :param state: Current financial analysis state
        :return: Updated financial analysis state with predictions
        """
        try:
            # Check if preprocessed data exists
            if not state.preprocessed_data:
                return state

            # Convert preprocessed data back to DataFrame
            data = pd.DataFrame.from_dict(state.preprocessed_data)

            # Prepare data for ARIMA
            close_prices = data['Close']

            # Fit ARIMA model
            model = ARIMA(close_prices, order=(5, 1, 0))
            model_fit = model.fit()

            # Generate forecast for next 7 days
            forecast = model_fit.forecast(steps=7)
            forecast_df = pd.DataFrame({
                'Predicted_Close': forecast,
                'Date': pd.date_range(start=close_prices.index[-1], periods=8)[1:]
            }).set_index('Date')

            # Update state with predictions
            state.predictions = forecast_df.to_dict()

            return state
        except Exception as e:
            state.error = str(e)
            return state

    def generate_report(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Generate comprehensive financial analysis report

        :param state: Current financial analysis state
        :return: Updated financial analysis state with report
        """
        try:
            # Check if predictions and preprocessed data exist
            if not state.predictions or not state.preprocessed_data:
                state.error = "Insufficient data for report generation"
                return state

            # Convert data back to DataFrame
            preprocessed_data = pd.DataFrame.from_dict(state.preprocessed_data)
            predictions = pd.DataFrame.from_dict(state.predictions)

            # Calculate key metrics
            last_close_price = float(preprocessed_data['Close'].iloc[-1])
            predicted_prices = predictions['Predicted_Close']

            # Calculate price change percentage
            price_change_pct = (
                predicted_prices.iloc[-1] - last_close_price) / last_close_price * 100
            volatility = float(preprocessed_data['Returns'].std() * 100)

            # Prepare predicted prices string
            predicted_prices_str = "\n".join([
                f"  {date.date()}: {price:.2f}"
                for date, price in predicted_prices.items()
            ])

            # Prepare context for Gemini
            context = f"""
            Comprehensive Stock Analysis Report

            Stock Symbol: {state.stock_symbol}
            Last Closing Price: {last_close_price:.2f}
            
            Predicted Price Trajectory:
            {predicted_prices_str}
            
            Key Insights:
            - Projected Price Change: {price_change_pct:.2f}%
            - Historical Volatility: {volatility:.2f}%
            
            Detailed Market Analysis:
            Provide a comprehensive analysis of the stock's potential movement, 
            including fundamental and technical insights. Consider:
            1. Current market trends
            2. Potential growth factors
            3. Risk assessment
            4. Short-term and long-term investment outlook
            """

            # Use Gemini for generating insights
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(context)

            # Update state with analysis report
            state.analysis_report = response.text

            return state
        except Exception as e:
            state.error = str(e)
            return state

    def create_visualization(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        """
        Create stock price forecast visualization

        :param state: Current financial analysis state
        :return: Updated financial analysis state with visualization path
        """
        try:
            # Check if data exists
            if not state.preprocessed_data or not state.predictions:
                state.error = "Insufficient data for visualization"
                return state

            # Convert data back to DataFrame
            preprocessed_data = pd.DataFrame.from_dict(state.preprocessed_data)
            predictions = pd.DataFrame.from_dict(state.predictions)

            # Create visualization directory if it doesn't exist
            os.makedirs('visualizations', exist_ok=True)

            plt.figure(figsize=(12, 6))

            # Plot historical prices
            plt.plot(preprocessed_data['Close'], label='Historical Prices')

            # Plot predictions
            plt.plot(predictions.index, predictions['Predicted_Close'],
                     color='red', label='Predicted Prices')

            plt.title(f'{state.stock_symbol} Price Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=45)

            # Save visualization
            visualization_path = f'visualizations/{state.stock_symbol}_forecast.png'
            plt.savefig(visualization_path, bbox_inches='tight')
            plt.close()

            # Update state with visualization path
            state.visualization_path = visualization_path

            return state
        except Exception as e:
            state.error = str(e)
            return state

    def comprehensive_analysis(self, stock_symbol: str, gemini_key: str) -> FinancialAnalysisState:
        """
        Perform comprehensive stock analysis

        :param stock_symbol: Stock symbol to analyze
        :param gemini_key: API key for Gemini
        :return: Comprehensive financial analysis state
        """
        # Set the Gemini API key
        genai.configure(api_key=gemini_key)

        # Fetch and preprocess data
        state = self.fetch_and_preprocess_data(stock_symbol)

        # If there's an error in initial data fetching, return immediately
        if state.error:
            return state

        # Generate predictions
        state = self.generate_predictions(state)

        # Generate report
        state = self.generate_report(state)

        # Create visualization
        state = self.create_visualization(state)

        return state
