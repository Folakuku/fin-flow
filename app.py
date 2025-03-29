# app.py

import pandas as pd
import yfinance as yf
from io import BytesIO
import streamlit as st
import speech_recognition as sr
from diagram import render_in_streamlit
from backend import (
    analyze_ticker, get_sp500_tickers, get_currency_pairs,
    show_historical_chart, generate_pdf_report  # Import the new function
)


st.set_page_config(page_title="🎙️ Fin-Flow - Market Advisor", layout="centered")

# === 📦 Cached Data ===
@st.cache_data
def get_cached_tickers():
    """Fetch and cache S&P 500 tickers."""
    return sorted(get_sp500_tickers())

@st.cache_data
def get_cached_currencies():
    """Fetch and cache currency pairs."""
    return sorted(get_currency_pairs())

@st.cache_data
def get_cached_history(ticker: str) -> pd.DataFrame:
    """Fetch and cache historical data for a ticker."""
    return yf.Ticker(ticker + "=X" if len(ticker) > 4 else ticker).history(period="1y")

# === 🎤 Voice Input ===
def transcribe_speech() -> str:
    """Transcribe speech input with start/stop control."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=30)
            return r.recognize_google(audio)
        except sr.WaitTimeoutError:
            return "Timed out waiting for speech."
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand your speech."
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"

# === 🚀 Main App ===
def main():
    st.title("🎙️ Fin-Flow: Market Chat + Agent Q&A")

    # Session state for history and speaking
    if "history" not in st.session_state:
        st.session_state.history = []
    if "is_speaking" not in st.session_state:
        st.session_state.is_speaking = False

    # UI Controls
    col1, col2 = st.columns([1, 1])
    with col1:
        chart_type = st.radio("📉 Chart Type", ["Line", "Candlestick", "Japanese Candlestick", "OHLC"], horizontal=True)
    with col2:
        audio_enabled = st.toggle("🔊 Enable Audio", value=True)
        include_metadata = st.checkbox("✅ Include chart metadata?")

    # Speak button in a separate row
    st.button("🎤 Speak" if not st.session_state.is_speaking else "🛑 Stop Speaking", key="speak_button", on_click=lambda: toggle_speaking())
    if st.session_state.is_speaking:
        st.info("🎙️ Listening... Click 'Stop Speaking' to end.")

    tickers = get_cached_tickers()
    currencies = get_cached_currencies()
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_stock = st.selectbox("📈 Pick Stock", options=[""] + tickers)
    with col2:
        selected_currency = st.selectbox("💱 Pick Currency", options=[""] + currencies)

    user_input = st.text_input("💬 Ask FinAI or type a stock/currency...", key="prompt")

    # Feedback input
    feedback = st.text_area("📝 Analyst Feedback (optional)", key="feedback")

    # Trimming parameters
    if audio_enabled:
        st.markdown("### ✂️ Trim Audio Report")
        col1, col2 = st.columns(2)
        with col1:
            start_sec = st.slider("Start Time (seconds)", min_value=0.0, max_value=60.0, value=0.0, step=0.1)
        with col2:
            end_sec = st.slider("End Time (seconds)", min_value=0.0, max_value=60.0, value=30.0, step=0.1)

    if st.button("🔍 Analyze or Ask"):
        # Prioritize dropdowns over text input
        ticker = selected_stock if selected_stock else selected_currency if selected_currency else user_input
        if not ticker:
            st.warning("Please enter a ticker or select a stock/currency.")
            return

        # Pass trimming parameters if audio is enabled
        result = analyze_ticker(
            ticker,
            feedback=feedback,
            audio_enabled=audio_enabled,
            start_sec=start_sec if audio_enabled else 0,
            end_sec=end_sec if audio_enabled else float('inf')
        )
        result["query"] = ticker
        st.session_state.history.append(result)

        # Display results
        st.markdown("### 📑 FinAI Report")
        st.markdown(result["response"])

        if result["viz_buffer"]:
            st.image(result["viz_buffer"], caption="📈 Forecasted Trend", use_container_width=True)
            st.download_button("💾 Download Forecast Chart", result["viz_buffer"], file_name="forecast_chart.png")

        historical_chart = None
        if result["ticker"]:
            historical_chart = show_historical_chart(result["ticker"], chart_type, include_metadata)
            st.plotly_chart(historical_chart, use_container_width=True)  # Use st.plotly_chart for interactivity
            # Still provide a download option for the static image
            buffer = BytesIO()
            historical_chart.write_image(buffer, format="png")
            st.download_button("💾 Download Historical Chart", buffer, file_name=f"{result['ticker']}_chart.png")

        if result["audio_path"]:
            st.audio(result["audio_path"])

        if result["report"]:
            st.download_button("🧾 Download Report (.txt)", result["report"], file_name="finai_report.txt")

        # Add PDF download button if report and charts are available
        if result["report"] and result["viz_buffer"] and historical_chart:
            pdf_buffer = generate_pdf_report(result["response"], result["viz_buffer"], historical_chart)
            st.download_button(
                "📜 Download Report as PDF",
                pdf_buffer,
                file_name="finai_report.pdf",
                mime="application/pdf"
            )

        if result["feedback"]:
            st.success(f"Feedback saved: {result['feedback']}")

    # Show history
    with st.expander("📚 View Past Analyses"):
        for past in reversed(st.session_state.history[-5:]):
            st.markdown(f"**🧍 Query:** {past['query'] or 'N/A'}")
            st.markdown(f"**🤖 Report:** {past['response'][:300]}...")
            if past['feedback']:
                st.markdown(f"**📝 Feedback:** {past['feedback']}")
            st.markdown("---")
    
    # 🔁 Diagram viewer (place this at the end)
    with st.expander("🧠 View FinFlow Diagrams"):
        render_in_streamlit()

def toggle_speaking():
    """Toggle the speaking state and handle transcription."""
    if not st.session_state.is_speaking:
        st.session_state.is_speaking = True
        spoken = transcribe_speech()
        st.session_state.prompt = spoken  # Populate the text input
        st.success(f"You said: `{spoken}`")
    st.session_state.is_speaking = False

if __name__ == "__main__":
    main()