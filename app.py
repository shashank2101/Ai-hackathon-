"""
AI Friday Season 2 — Data Analytics Automated Report Narrative Generator
Streamlit UI
"""

import os
import sys
import json
import time
import io

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_samples.sample_data import SAMPLE_DATASETS, get_sample_dataset
from agents.pipeline import run_pipeline
from mcp_server.tool_executor import (
    kpi_summarizer, trend_analyzer, anomaly_detector,
    forecast_analyzer, correlation_analyzer
)

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataNarrator AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; color: #e94560; }
    .main-header p { color: #a8b2d8; margin-top: 0.5rem; font-size: 1rem; }

    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .agent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .narrative-box {
        background: #1e1e2e;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.8;
        color: #cdd6f4;
        font-size: 0.95rem;
        white-space: pre-wrap;
    }
    .step-indicator {
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background: #313244;
        border-radius: 8px;
        margin: 0.3rem 0;
        color: #cdd6f4;
        font-size: 0.85rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #0f3460);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 DataNarrator AI</h1>
    <p>AI Friday Season 2 · Data Analytics Automated Report Narrative Generator<br>
    <small>Powered by Gemini · LangGraph · MCP Tools</small></p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────
api_key = os.getenv("GEMINI_API_KEY", "")

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    if api_key:
        st.success("✅ Gemini API key loaded from .env")
    else:
        st.error("❌ GEMINI_API_KEY not found in .env file")
        st.caption("Add it to your `.env` file:\n```\nGEMINI_API_KEY=AIza...\n```")

    st.markdown("---")
    st.markdown("### 📂 Data Source")
    data_source = st.radio("Choose input", ["📁 Upload File", "🎲 Sample Dataset"])

    df = None
    if data_source == "📁 Upload File":
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_json(uploaded)
                st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        sample_name = st.selectbox("Choose dataset", list(SAMPLE_DATASETS.keys()))
        if st.button("Load Sample"):
            df = get_sample_dataset(sample_name)
            st.session_state["df"] = df
            st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    if "df" in st.session_state and df is None:
        df = st.session_state["df"]

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
```
Input Dataset
    ↓
Guardrail + LLM
    ↓
Router Agent
    ↓
┌──────────────┐
│ Sales Agent  │
│ Finance Agent│
│ Customer     │
│ Operations   │
└──────────────┘
    ↓
MCP Tools
(Trend/Anomaly/
 Forecast/KPI/
 Correlation)
    ↓
Output Guardrail
+ Final LLM
    ↓
📄 Narrative Report
```
""")


# ── Main Content ─────────────────────────────────────────────────────────────
if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Data Preview", "📈 Visual Analysis", "🤖 AI Narrative", "🔍 Tool Insights"]
    )

    # ── Tab 1: Data Preview ────────────────────────────────────────────────
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            st.metric("Numeric Cols", len(num_cols))
        with col4:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        st.markdown("#### Dataset Preview")
        st.dataframe(df, use_container_width=True, height=300)

        st.markdown("#### Column Types")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count().values,
            "Unique": df.nunique().values
        })
        st.dataframe(dtype_df, use_container_width=True)

    # ── Tab 2: Visual Analysis ─────────────────────────────────────────────
    with tab2:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.warning("No numeric columns found for visualization.")
        else:
            st.markdown("#### 📊 Numeric Column Distributions")
            n = len(num_cols)
            cols_per_row = 2
            rows = (n + cols_per_row - 1) // cols_per_row

            for i in range(0, n, cols_per_row):
                viz_cols = st.columns(cols_per_row)
                for j, col_name in enumerate(num_cols[i:i+cols_per_row]):
                    with viz_cols[j]:
                        fig = px.line(
                            df, y=col_name,
                            title=f"{col_name} Over Time",
                            template="plotly_dark",
                            color_discrete_sequence=["#e94560"]
                        )
                        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig, use_container_width=True)

            if len(num_cols) >= 2:
                st.markdown("#### 🔗 Correlation Heatmap")
                corr_matrix = df[num_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale="RdBu_r",
                    template="plotly_dark",
                    title="Pearson Correlation Matrix",
                    zmin=-1, zmax=1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: AI Narrative ────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🤖 Generate AI Narrative Report")

        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar to generate narratives.")
        else:
            if st.button("🚀 Generate Report", key="gen_btn"):
                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                steps = [
                    ("🛡️", "Input Guardrail: Validating dataset quality..."),
                    ("🔀", "Router Agent: Detecting dataset type and routing to specialist..."),
                    ("🔧", "MCP Tools: Running Trend, Anomaly, Forecast, KPI, Correlation analysis..."),
                    ("🤖", "Specialist Agent: Generating domain-specific insights..."),
                    ("✨", "Output Guardrail: Polishing final narrative..."),
                ]

                with st.spinner("Running multi-agent pipeline..."):
                    # Show animated steps
                    step_box = st.empty()
                    for icon, msg in steps:
                        step_box.markdown(f'<div class="step-indicator">{icon} {msg}</div>', unsafe_allow_html=True)
                        time.sleep(0.5)

                    start_time = time.time()
                    try:
                        result = run_pipeline(df, api_key)
                        elapsed = time.time() - start_time

                        step_box.empty()

                        if result.get("error"):
                            st.error(f"❌ {result['error']}")
                        else:
                            st.session_state["pipeline_result"] = result
                            st.session_state["gen_time"] = elapsed

                            # Success metrics
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("⏱️ Generation Time", f"{elapsed:.1f}s")
                            with c2:
                                dtype = result.get("dataset_type", "general").title()
                                st.metric("🤖 Agent Used", f"{dtype} Agent")
                            with c3:
                                words = len(result.get("final_narrative", "").split())
                                st.metric("📝 Report Words", words)

                    except Exception as e:
                        step_box.empty()
                        st.error(f"Pipeline error: {str(e)}")
                        st.exception(e)

            # Display result if available
            if "pipeline_result" in st.session_state:
                result = st.session_state["pipeline_result"]
                narrative = result.get("final_narrative", "")

                if narrative:
                    st.markdown("---")
                    agent_type = result.get("dataset_type", "general").title()
                    st.markdown(f"**Agent:** `{agent_type} Agent` | **Tools Used:** Trend Analyzer · Anomaly Detector · Forecast Analyzer · KPI Summarizer · Correlation Analyzer")

                    st.markdown("#### 📄 Generated Narrative Report")
                    st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

                    # Download button
                    st.download_button(
                        label="⬇️ Download Report (TXT)",
                        data=narrative,
                        file_name="analytics_narrative_report.txt",
                        mime="text/plain"
                    )

    # ── Tab 4: Tool Insights ───────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔍 MCP Tool Outputs")
        st.markdown("Run individual analytics tools to inspect raw results.")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        df_json = df.to_json(orient="records")

        t1, t2, t3, t4, t5 = st.tabs([
            "📈 Trends", "⚠️ Anomalies", "🔮 Forecast", "📊 KPIs", "🔗 Correlations"
        ])

        with t1:
            if st.button("Run Trend Analyzer"):
                with st.spinner("Analyzing trends..."):
                    trends = trend_analyzer(df_json, num_cols)
                st.json(trends)
                for col, info in trends.items():
                    direction_emoji = "⬆️" if info["direction"] == "upward" else ("⬇️" if info["direction"] == "downward" else "➡️")
                    st.markdown(f"{direction_emoji} **{col}**: {info['summary']}")

        with t2:
            if st.button("Run Anomaly Detector"):
                with st.spinner("Detecting anomalies..."):
                    anomalies = anomaly_detector(df_json, num_cols)
                st.json(anomalies)
                for col, info in anomalies.items():
                    sev_color = "🔴" if info["severity"] == "high" else ("🟡" if info["severity"] == "medium" else "🟢")
                    st.markdown(f"{sev_color} **{col}**: {info['summary']}")

        with t3:
            if num_cols:
                target = st.selectbox("Select column to forecast", num_cols)
                periods = st.slider("Periods ahead", 1, 10, 3)
                if st.button("Run Forecast"):
                    with st.spinner("Forecasting..."):
                        fc = forecast_analyzer(df_json, target, periods)
                    st.json(fc)

                    # Visualize forecast
                    series = df[target].dropna().values
                    x_hist = list(range(len(series)))
                    x_fore = list(range(len(series), len(series) + periods))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_hist, y=series, name="Historical", line=dict(color="#e94560")))
                    fig.add_trace(go.Scatter(x=x_fore, y=fc["linear_forecasts"], name="Linear Forecast",
                                             line=dict(color="#89b4fa", dash="dash")))
                    fig.add_trace(go.Scatter(x=x_fore, y=fc["moving_avg_forecasts"], name="MA Forecast",
                                             line=dict(color="#a6e3a1", dash="dot")))
                    fig.update_layout(template="plotly_dark", title=f"Forecast: {target}", height=350)
                    st.plotly_chart(fig, use_container_width=True)

        with t4:
            if st.button("Run KPI Summarizer"):
                with st.spinner("Computing KPIs..."):
                    kpis = kpi_summarizer(df_json, num_cols)
                kpi_rows = []
                for col, info in kpis.items():
                    kpi_rows.append({
                        "Column": col,
                        "Mean": info["mean"],
                        "Median": info["median"],
                        "Std Dev": info["std_dev"],
                        "Min": info["min"],
                        "Max": info["max"],
                        "Growth %": info["growth_rate_pct"]
                    })
                if kpi_rows:
                    st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True)

        with t5:
            if len(num_cols) >= 2:
                thresh = st.slider("Correlation threshold", 0.3, 1.0, 0.7)
                if st.button("Run Correlation Analyzer"):
                    with st.spinner("Analyzing correlations..."):
                        corr_result = correlation_analyzer(df_json, num_cols, thresh)
                    st.markdown(f"**{corr_result['summary']}**")
                    if corr_result["strong_correlations"]:
                        st.json(corr_result["strong_correlations"])
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #a8b2d8;">
        <h2>👈 Get Started</h2>
        <p>Enter your Gemini API key and upload a dataset or choose a sample from the sidebar.</p>
        <br>
        <h4>🏗️ What this does</h4>
        <p>Automatically converts your CSV/JSON dataset into a professional business narrative report
        using a multi-agent LangGraph pipeline with specialized domain agents and MCP analytics tools.</p>
        <br>
        <h4>✨ Features</h4>
        <p>
        📊 Trend Detection &nbsp;|&nbsp; ⚠️ Anomaly Detection &nbsp;|&nbsp; 🔮 Forecasting<br>
        📈 KPI Summarization &nbsp;|&nbsp; 🔗 Correlation Analysis<br>
        🤖 Domain-Specific Agents (Sales / Finance / Customer / Operations)<br>
        📄 Auto-generated Business Narrative Reports
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#585b70; font-size:0.8rem;'>"
    "AI Friday Season 2 · DataNarrator AI · Built with LangGraph + Gemini + MCP"
    "</div>",
    unsafe_allow_html=True
)
