# diagram.py

import os, time
from graphviz import Source
from backend import fin_graph_compiled
from langgraph.graph import StateGraph, START, END
from langgraph.graph.visualization import draw_graph, draw_mermaid

# === COMBINED ROUTING LOGIC ===
def combined_graph():
    g = StateGraph(dict)
    g.add_node("route_input", lambda x: x)
    g.add_node("basic_analysis", lambda x: x)
    g.add_node("financial_analysis", lambda x: x)
    g.add_node("merge_results", lambda x: x)

    g.add_edge(START, "route_input")
    g.add_conditional_edges(
        "route_input",
        lambda state: "basic_analysis" if state.get("quick") else "financial_analysis",
        {
            "basic_analysis": "merge_results",
            "financial_analysis": "merge_results"
        }
    )
    g.add_edge("merge_results", END)
    return g.compile()


# === STATIC SAVERS ===
def save_combined_route_diagram():
    print("📊 Generating combined LangGraph diagram...")
    g = combined_graph()
    dot = draw_graph(g, show=False)

    # Labels
    dot.node("route_input", "🔀 Route Input\nInput: Query\nOutput: Path Flag")
    dot.node("basic_analysis", "🧠 Basic LLM Analysis\nInput: Quick Query\nOutput: Draft Response")
    dot.node("financial_analysis", "📊 Full FinAI Pipeline\nInput: Detailed Query\nOutput: Rich Report")
    dot.node("merge_results", "🔗 Merge Results\nInputs: basic/fin\nOutput: Unified Dict")
    dot.node("__start__", "🚀 Start", shape="oval", style="filled", fillcolor="#B3E5FC")
    dot.node("__end__", "🏁 End", shape="oval", style="filled", fillcolor="#FFCC80")

    os.makedirs("images", exist_ok=True)
    dot.render("images/combined_flow", format="png", cleanup=True)
    print("✅ Saved: images/combined_flow.png")


def save_finflow_pipeline_diagram():
    print("🧠 Generating FinFlow pipeline diagram...")
    dot = draw_graph(fin_graph_compiled, show=False)

    dot.node("data_ingestion", "📥 Data Ingestion\nSource: yfinance, Alpha Vantage")
    dot.node("market_analysis", "📊 Forecasting\nTool: Auto-ARIMA")
    dot.node("anomaly_detection", "📉 Anomaly Detection\nMethod: z-score")
    dot.node("report_generation", "🧠 Report Gen\nAgent: FinAI + Tavily + LLM")
    dot.node("generate_audio", "🔊 Audio Summary\nTool: gTTS + pydub")

    os.makedirs("images", exist_ok=True)
    dot.render("images/finflow_pipeline", format="png", cleanup=True)
    print("✅ Saved: images/finflow_pipeline.png")


def save_mermaid_diagram():
    print("📄 Exporting Mermaid + SVG...")
    mermaid_code = draw_mermaid(fin_graph_compiled)
    os.makedirs("docs", exist_ok=True)

    with open("docs/langgraph.md", "w") as f:
        f.write("```mermaid\n" + mermaid_code + "\n```")
    print("✅ Saved: docs/langgraph.md")

    dot = draw_graph(fin_graph_compiled, show=False)
    svg = Source(dot.source).pipe(format='svg')
    with open("docs/finflow_pipeline.svg", "wb") as f:
        f.write(svg)
    print("✅ Saved: docs/finflow_pipeline.svg")


# === AUTOLABEL FROM LANGCHAIN CHAINS ===
def extract_node_labels_from_chain(chain) -> dict:
    from langchain_core.chains.base import Chain
    if not isinstance(chain, Chain): return {}
    return {
        name: f"🤖 {name}\n{step.prompt.template[:80]}..."
        for name, step in getattr(chain, "steps", {}).items()
        if hasattr(step, "prompt")
    }


# === STREAMLIT EMBED + DEV REFRESH ===
def get_labeled_graph(graph, mode="graphviz"):
    if mode == "graphviz":
        dot = draw_graph(graph, show=False)
        dot.node("data_ingestion", "📥 Data Ingestion\nFrom: yfinance / AlphaVantage")
        dot.node("market_analysis", "📈 Forecasting\nModel: AutoARIMA")
        dot.node("anomaly_detection", "⚠️ Anomaly Detection\nMetric: z-score")
        dot.node("report_generation", "📝 Report & Chart Gen\nLLM + gTTS + Plotly")
        dot.node("generate_audio", "🔊 Audio Report\nLibs: gTTS, pydub")
        return dot
    elif mode == "mermaid":
        return draw_mermaid(graph)


def render_in_streamlit():
    import streamlit as st
    st.markdown("### 🧠 FinFlow Pipeline Visualizer")

    view = st.radio("View mode", ["Graphviz", "Mermaid"], horizontal=True)
    graph_type = st.selectbox("Pipeline", ["🔗 Combined Flow", "📊 FinAI Flow"])
    graph = combined_graph() if graph_type == "🔗 Combined Flow" else fin_graph_compiled
    mode = "graphviz" if view == "Graphviz" else "mermaid"

    code_file = "backend.py"
    last_mod_time = os.path.getmtime(code_file)
    if st.session_state.get("last_rendered_graph") != last_mod_time:
        st.session_state["last_rendered_graph"] = last_mod_time

    if mode == "graphviz":
        dot = get_labeled_graph(graph, "graphviz")
        st.graphviz_chart(dot.source)
    else:
        mermaid = get_labeled_graph(graph, "mermaid")
        st.code("```mermaid\n" + mermaid + "\n```", language="markdown")

    st.caption(f"🕒 Last refreshed: {time.ctime(last_mod_time)}")


# === MAIN CLI ENTRYPOINT ===
if __name__ == "__main__":
    save_combined_route_diagram()
    save_finflow_pipeline_diagram()
    save_mermaid_diagram()
    print("🏁 All diagrams generated.")
