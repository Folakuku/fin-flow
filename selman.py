import os
import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gtts import gTTS
import tempfile
import google.generativeai as genai
from statsmodels.tsa.arima.model import ARIMA
from typing import TypedDict, List, Optional
import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
from dotenv import load_dotenv
import os

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")


# Setup Gemini API
genai.configure(api_key=GENAI_API_KEY)


class FinancialAnalysisState(TypedDict):
    """
    State management for the financial analysis workflow
    """
    stock_symbol: str
    raw_data: Optional[pd.DataFrame]
    preprocessed_data: Optional[pd.DataFrame]
    predictions: Optional[pd.DataFrame]
    analysis_report: Optional[str]
    visualization_path: Optional[str]
    news: Optional[List[str]]


class FinancialAnalysisSystem:
    def __init__(self):
        """
        Initialize the Financial Analysis System components
        """
        # Web Scraping Component
        self.scraper = WebScraper()

        # Create LangGraph Workflow
        self.workflow = self.create_workflow()

    def create_data_ingestion_node(self):
        """
        Create data ingestion and preprocessing node
        """
        def data_ingestion(state: FinancialAnalysisState):
            try:
                # Fetch stock data from Yahoo Finance
                stock_data = yf.download(state['stock_symbol'],
                                         period='1mo',
                                         interval='1d')

                # Preprocess data
                preprocessed_data = self.preprocess_data(stock_data)

                return {
                    **state,
                    'raw_data': stock_data,
                    'preprocessed_data': preprocessed_data
                }
            except Exception as e:
                print(f"Data ingestion error: {e}")
                return {
                    **state,
                    'raw_data': None,
                    'preprocessed_data': None,
                    'error': str(e)
                }
        return data_ingestion

    def preprocess_data(self, data):
        """
        Preprocess stock market data
        """
        # Basic preprocessing steps
        if data is None or data.empty:
            return None

        data = data.dropna()
        data.index = pd.to_datetime(data.index)  # Ensure datetime index
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(1 + data['Returns'])
        return data

    def create_predictive_modeling_node(self):
        """
        Create predictive modeling node using ARIMA
        """
        def predictive_modeling(state: FinancialAnalysisState):
            try:
                # Check if preprocessed data exists
                if state.get('preprocessed_data') is None:
                    return {
                        **state,
                        'predictions': None,
                        'error': 'No preprocessed data available'
                    }

                # Prepare data for ARIMA
                data = state['preprocessed_data']['Close']

                # Fit ARIMA model
                model = ARIMA(data, order=(5, 1, 0))
                model_fit = model.fit()

                # Generate forecast for next 7 days
                forecast = model_fit.forecast(steps=7)
                forecast_df = pd.DataFrame({
                    'Predicted_Close': forecast,
                    'Date': pd.date_range(start=data.index[-1], periods=8)[1:]
                }).set_index('Date')

                return {
                    **state,
                    'predictions': forecast_df
                }
            except Exception as e:
                print(f"Predictive modeling error: {e}")
                return {
                    **state,
                    'predictions': None,
                    'error': str(e)
                }
        return predictive_modeling

    def create_report_generation_node(self):
        """
        Create report generation node using Gemini
        """
        def report_generation(state: FinancialAnalysisState):
            try:
                # Check if predictions exist
                if state.get('predictions') is None or state.get('preprocessed_data') is None:
                    return {
                        **state,
                        'analysis_report': 'Unable to generate report due to missing data',
                        'error': 'Missing predictions or preprocessed data'
                    }

                # Prepare context for analysis
                last_close_price = float(
                    state['preprocessed_data']['Close'].iloc[-1])
                predicted_prices = state['predictions']['Predicted_Close']

                # Calculate key metrics
                price_change_pct = (
                    predicted_prices.iloc[-1] - last_close_price) / last_close_price * 100
                volatility = float(
                    state['preprocessed_data']['Returns'].std() * 100)

                # Convert predicted prices to a formatted string
                predicted_prices_str = "\n".join([
                    f"  {date.date()}: {price:.2f}"
                    for date, price in predicted_prices.items()
                ])

                context = f"""
                Comprehensive Stock Analysis Report

                Stock Symbol: {state['stock_symbol']}
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

                return {
                    **state,
                    'analysis_report': response.text
                }
            except Exception as e:
                print(f"Report generation error: {e}")
                return {
                    **state,
                    'analysis_report': f'Error generating report: {str(e)}',
                    'error': str(e)
                }
        return report_generation

    def create_visualization_node(self):
        """
        Create visualization node
        """
        def create_visualization(state: FinancialAnalysisState):
            try:
                # Check if data exists
                if state.get('preprocessed_data') is None or state.get('predictions') is None:
                    return {
                        **state,
                        'visualization_path': None,
                        'error': 'Insufficient data for visualization'
                    }

                plt.figure(figsize=(12, 6))

                # Plot historical prices
                plt.plot(state['preprocessed_data']['Close'],
                         label='Historical Prices')

                # Plot predictions
                predictions = state['predictions']
                plt.plot(predictions.index, predictions['Predicted_Close'],
                         color='red', label='Predicted Prices')

                plt.title(f'{state["stock_symbol"]} Price Forecast')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.xticks(rotation=45)

                # Save visualization
                visualization_path = f'{state["stock_symbol"]}_forecast.png'
                plt.savefig(visualization_path, bbox_inches='tight')
                plt.close()

                return {
                    **state,
                    'visualization_path': visualization_path
                }
            except Exception as e:
                print(f"Visualization error: {e}")
                return {
                    **state,
                    'visualization_path': None,
                    'error': str(e)
                }
        return create_visualization

    def create_workflow(self):
        """
        Create workflow steps
        """
        def workflow(stock_symbol):
            # Simulate workflow steps
            try:
                # Initial state
                state = {
                    'stock_symbol': stock_symbol,
                    'raw_data': None,
                    'preprocessed_data': None,
                    'predictions': None,
                    'analysis_report': None,
                    'visualization_path': None,
                    'news': None
                }

                # Data Ingestion
                data_ingestion_node = self.create_data_ingestion_node()
                state = data_ingestion_node(state)

                # Predictive Modeling
                predictive_modeling_node = self.create_predictive_modeling_node()
                state = predictive_modeling_node(state)

                # Report Generation
                report_generation_node = self.create_report_generation_node()
                state = report_generation_node(state)

                # Visualization
                visualization_node = self.create_visualization_node()
                state = visualization_node(state)

                return state
            except Exception as e:
                return {'error': str(e)}

        return workflow


class WebScraper:
    """
    Web scraping component for financial news
    """

    def scrape_financial_news(self, stock_symbol):
        """
        Scrape financial news related to the stock
        """
        url = f'https://finance.yahoo.com/quote/{stock_symbol}'
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract relevant news (this is a basic implementation)
            news_elements = soup.find_all('h3', class_='Mb(5px)')
            news = [elem.get_text() for elem in news_elements[:5]]
            return news
        except Exception as e:
            print(f"Error scraping news: {e}")
            return []


class VoiceInteractionSystem:
    """
    Voice interaction system using GTTS
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def text_to_speech(self, text):
        """
        Convert text to speech
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_audio.name)
            return temp_audio.name

    def speech_to_text(self, audio_file):
        """
        Convert speech to text
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return "Sorry, I couldn't understand that."


class FinancialAnalystChatbot:
    """
    Interactive chatbot for financial analysis insights
    """

    def __init__(self):
        """
        Initialize the chatbot with Gemini Pro
        """
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.current_report = ""

    def set_current_report(self, report):
        """
        Set the current financial report for context
        """
        self.current_report = report

    def generate_response(self, message, history):
        """
        Generate a response based on the user's message and current financial report
        """
        try:
            # Prepare context with the financial report
            context = f"""
            You are a financial analyst assistant. 
            Current Financial Report Context:
            {self.current_report}

            User Question: {message}

            Please provide a detailed, professional response that:
            1. Relates the question to the current financial report
            2. Provides insights based on the available data
            3. Offers clear, actionable financial advice
            """

            # Generate response using Gemini
            response = self.model.generate_content(context)

            # Update history with the conversation
            updated_history = history + [[message, response.text]]

            return updated_history
        except Exception as e:
            error_response = f"Error generating response: {str(e)}"
            return history + [[message, error_response]]


class FeedbackSystem:
    """
    System for collecting and processing user feedback
    """

    def __init__(self):
        self.feedback_file = "user_feedback.csv"

        # Initialize feedback file if it doesn't exist
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                f.write("timestamp,stock_symbol,rating,comments\n")

    def save_feedback(self, stock_symbol, rating, comments):
        """
        Save user feedback to CSV file
        """
        try:
            timestamp = pd.Timestamp.now()
            with open(self.feedback_file, 'a') as f:
                f.write(f"{timestamp},{stock_symbol},{rating},{comments}\n")
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False


def create_financial_analysis_interface():
    """
    Create Gradio interface for Financial Analysis System
    """
    # Initialize system components
    analysis_system = FinancialAnalysisSystem()
    voice_system = VoiceInteractionSystem()
    web_scraper = WebScraper()
    chatbot = FinancialAnalystChatbot()
    feedback_system = FeedbackSystem()

    # List of popular stocks for dropdown
    popular_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "WMT",
        "JNJ", "PG", "DIS", "NFLX", "INTC"
    ]

    def process_analysis(stock_symbol):
        """
        Process financial analysis and generate output
        """
        try:
            # Run workflow
            final_state = analysis_system.workflow(stock_symbol)

            # Check for errors
            if final_state.get('error'):
                return (
                    None,
                    f"An error occurred: {final_state['error']}",
                    None,
                    [("System", f"An error occurred: {final_state['error']}")]
                )

            # Scrape news
            news = web_scraper.scrape_financial_news(stock_symbol)

            # Combine analysis and news
            full_report = f"""
            Stock Analysis Report for {stock_symbol}

            Predictive Analysis:
            {final_state.get('analysis_report', 'No analysis available')}

            Recent News:
            {chr(10).join(news) if news else 'No recent news found'}
            """

            # Set the report for the chatbot
            chatbot.set_current_report(full_report)

            # Generate audio
            audio_path = voice_system.text_to_speech(full_report)

            return (
                final_state.get('visualization_path'),
                full_report,
                audio_path,
                [["System", full_report]]  # Ensure list of lists format
            )

        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            return (None, error_message, None, [["System", error_message]])

    def process_voice_input(audio_file):
        """
        Process voice input and return transcribed text
        """
        try:
            text = voice_system.speech_to_text(audio_file)
            return text
        except Exception as e:
            return f"Error processing voice input: {str(e)}"

    def submit_feedback(stock_symbol, rating, comments):
        """
        Submit user feedback
        """
        success = feedback_system.save_feedback(stock_symbol, rating, comments)
        if success:
            return "Thank you for your feedback! It helps us improve our analysis."
        else:
            return "Sorry, there was an error saving your feedback. Please try again."

    # Create Gradio interface
    with gr.Blocks() as iface:
        # Stock Analysis Tab
        with gr.Tab("Stock Analysis"):
            with gr.Row():
                stock_dropdown = gr.Dropdown(
                    choices=popular_stocks, label="Select Popular Stock")
                stock_input = gr.Textbox(
                    label="Or Enter Stock Symbol (e.g., AAPL)")
                analyze_btn = gr.Button("Analyze")

            # Voice Input for Stock Selection
            with gr.Row():
                voice_input = gr.Audio(
                    type="filepath", label="Speak Stock Symbol")
                transcribe_btn = gr.Button("Transcribe")

            visualization = gr.Image(label="Price Forecast Visualization")
            report = gr.Textbox(label="Analysis Report")
            voice_report = gr.Audio(label="Voice Report")

            # Analysis Chat History
            analysis_chat = gr.Chatbot(label="Analysis Chat")

            # Connect dropdown to input field
            def update_stock_input(selection):
                return selection

            stock_dropdown.change(
                fn=update_stock_input,
                inputs=stock_dropdown,
                outputs=stock_input
            )

            # Connect voice transcription to input field
            transcribe_btn.click(
                fn=process_voice_input,
                inputs=voice_input,
                outputs=stock_input
            )

            # Bind analysis function
            analyze_btn.click(
                fn=process_analysis,
                inputs=stock_input,
                outputs=[visualization, report, voice_report, analysis_chat]
            )

        # Interactive Chat Tab
        with gr.Tab("Chat with Financial Analyst"):
            chatbot_component = gr.Chatbot(label="Financial Analyst")
            with gr.Row():
                msg = gr.Textbox(label="Your Question")
                voice_question = gr.Audio(
                    type="filepath", label="Speak Your Question")

            with gr.Row():
                submit_btn = gr.Button("Send")
                transcribe_question_btn = gr.Button("Transcribe Question")

            # Transcribe spoken question
            transcribe_question_btn.click(
                fn=process_voice_input,
                inputs=voice_question,
                outputs=msg
            )

            # Bind chat function
            submit_btn.click(
                fn=chatbot.generate_response,
                inputs=[msg, chatbot_component],
                outputs=[chatbot_component]
            ).then(
                fn=lambda x: "",
                inputs=msg,
                outputs=msg
            )

        # Feedback Tab
        with gr.Tab("Provide Feedback"):
            feedback_stock = gr.Textbox(label="Stock Symbol")
            feedback_rating = gr.Slider(
                minimum=1, maximum=5, step=1, label="Rating (1-5)")
            feedback_comments = gr.Textbox(label="Comments", lines=5)
            feedback_btn = gr.Button("Submit Feedback")
            feedback_result = gr.Textbox(label="Result")

            # Bind feedback function
            feedback_btn.click(
                fn=submit_feedback,
                inputs=[feedback_stock, feedback_rating, feedback_comments],
                outputs=feedback_result
            )

    return iface


# Launch the interface
if __name__ == "__main__":
    interface = create_financial_analysis_interface()
    interface.launch(share=True)
