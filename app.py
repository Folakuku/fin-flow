# app.py
import streamlit as st
from backend import run_analysis, get_sp500_tickers

st.set_page_config(page_title="FinAI", layout="centered")

def main():
    st.title("🎙️ FinAI - Voice-Enabled Market Advisor")

    tickers = get_sp500_tickers()
    ticker = st.selectbox("Choose Stock Ticker", options=tickers)
    chart_type = st.selectbox("Choose Chart Type", ["line", "candlestick", "japanese"])
    audio_enabled = st.checkbox("🔊 Enable Audio Report", value=True)

    if st.button("Analyze"):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                result = run_analysis(ticker=ticker, chart_type=chart_type)

                st.subheader("📋 AI Report")
                st.write(result["report"])

                st.subheader("📊 Chart")
                st.image(result["viz_buffer"], caption=f"{ticker} Forecast")

                if audio_enabled:
                    st.subheader("🔈 Audio Summary")
                    st.audio(result["audio_path"])

            except Exception as e:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
