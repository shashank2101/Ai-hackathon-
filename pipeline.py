"""
LangGraph Multi-Agent Pipeline
Architecture:
  Input → Guardrail+BasicLLM → Router Agent → [Sales|Financial|Customer|Operations] Agent
       → MCP Tools → Output Guardrail + Final Insights LLM
"""

import json
import os
from typing import TypedDict, Annotated, Literal
import operator

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from mcp_server.tool_executor import (
    trend_analyzer, anomaly_detector, forecast_analyzer,
    kpi_summarizer, correlation_analyzer
)
from agents.guardrails import (
    validate_input, detect_dataset_type,
    validate_narrative, sanitize_output
)


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
class AnalyticsState(TypedDict):
    # Input
    df_json: str
    df_summary: str
    dataset_type: str
    num_cols: list[str]
    all_cols: list[str]

    # Intermediate tool outputs
    kpi_results: dict
    trend_results: dict
    anomaly_results: dict
    forecast_results: dict
    correlation_results: dict

    # Agent-specific narrative chunks
    agent_narrative: str

    # Final output
    final_narrative: str
    error: str


# ─────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────
def get_llm(api_key: str, temperature: float = 0.3):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=temperature,
    )


# ─────────────────────────────────────────────
# Helper: call LLM
# ─────────────────────────────────────────────
def _llm_call(llm, system_prompt: str, user_prompt: str) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


# ─────────────────────────────────────────────
# Node 1: Input Guardrail + Schema Extraction
# ─────────────────────────────────────────────
def input_guardrail_node(state: AnalyticsState) -> AnalyticsState:
    df = pd.read_json(state["df_json"], orient="records")
    is_valid, msg = validate_input(df)
    if not is_valid:
        return {**state, "error": msg}

    dataset_type = detect_dataset_type(df)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    # Basic schema summary
    df_summary = (
        f"Rows: {len(df)}, Columns: {all_cols}\n"
        f"Numeric columns: {num_cols}\n"
        f"Null counts: {df.isnull().sum().to_dict()}\n"
        f"Sample (first 3 rows):\n{df.head(3).to_dict(orient='records')}"
    )

    return {
        **state,
        "dataset_type": dataset_type,
        "num_cols": num_cols,
        "all_cols": all_cols,
        "df_summary": df_summary,
        "error": "",
    }


# ─────────────────────────────────────────────
# Node 2: Router Agent
# ─────────────────────────────────────────────
def router_node(state: AnalyticsState) -> AnalyticsState:
    """Passes schema to LLM for routing decision — embedded in dataset_type."""
    # Already determined in guardrail; no extra LLM call needed for routing
    # (Can be extended for ambiguous cases)
    return state


def route_to_agent(state: AnalyticsState) -> Literal["sales_agent", "financial_agent", "customer_agent", "operations_agent"]:
    dtype = state.get("dataset_type", "general")
    mapping = {
        "sales": "sales_agent",
        "financial": "financial_agent",
        "customer": "customer_agent",
        "operations": "operations_agent",
        "general": "sales_agent",  # default
    }
    return mapping.get(dtype, "sales_agent")


# ─────────────────────────────────────────────
# Shared: Run MCP Tools
# ─────────────────────────────────────────────
def run_mcp_tools(state: AnalyticsState) -> AnalyticsState:
    df_json = state["df_json"]
    num_cols = state["num_cols"]

    kpi = kpi_summarizer(df_json, num_cols)
    trend = trend_analyzer(df_json, num_cols)
    anomaly = anomaly_detector(df_json, num_cols)
    corr = correlation_analyzer(df_json, num_cols)

    # Forecast on the first numeric column
    forecast = {}
    if num_cols:
        try:
            forecast = forecast_analyzer(df_json, num_cols[0], periods_ahead=3)
        except Exception:
            forecast = {}

    return {
        **state,
        "kpi_results": kpi,
        "trend_results": trend,
        "anomaly_results": anomaly,
        "forecast_results": forecast,
        "correlation_results": corr,
    }


# ─────────────────────────────────────────────
# Agent Nodes (each has domain-specific prompt)
# ─────────────────────────────────────────────
def _build_tool_context(state: AnalyticsState) -> str:
    return f"""
KPIs: {json.dumps(state.get('kpi_results', {}), indent=2)}
Trends: {json.dumps(state.get('trend_results', {}), indent=2)}
Anomalies: {json.dumps(state.get('anomaly_results', {}), indent=2)}
Forecast: {json.dumps(state.get('forecast_results', {}), indent=2)}
Correlations: {json.dumps(state.get('correlation_results', {}), indent=2)}
"""


def sales_agent_node(state: AnalyticsState, api_key: str) -> AnalyticsState:
    llm = get_llm(api_key)
    system = """You are a Senior Sales Analytics expert. 
Analyze the provided sales data metrics and write a clear, business-friendly narrative report.
Focus on: revenue trends, top/bottom performers, growth rates, sales velocity, anomalies in orders.
Structure: Executive Summary → Key Findings → Anomalies → Recommendations.
Write for a business stakeholder — avoid jargon, use plain English."""
    user = f"""Dataset Schema:\n{state['df_summary']}\n\nAnalysis Results:\n{_build_tool_context(state)}
Write a complete sales analytics narrative report."""
    narrative = _llm_call(llm, system, user)
    return {**state, "agent_narrative": narrative}


def financial_agent_node(state: AnalyticsState, api_key: str) -> AnalyticsState:
    llm = get_llm(api_key)
    system = """You are a Senior Financial Analyst.
Analyze the financial metrics and write a clear executive narrative.
Focus on: P&L trends, cost analysis, budget variances, profit margins, financial anomalies.
Structure: Executive Summary → Financial Highlights → Risk Indicators → Recommendations.
Write concisely for C-suite stakeholders."""
    user = f"""Dataset Schema:\n{state['df_summary']}\n\nAnalysis Results:\n{_build_tool_context(state)}
Write a complete financial analytics narrative report."""
    narrative = _llm_call(llm, system, user)
    return {**state, "agent_narrative": narrative}


def customer_agent_node(state: AnalyticsState, api_key: str) -> AnalyticsState:
    llm = get_llm(api_key)
    system = """You are a Customer Analytics Expert.
Analyze customer metrics and write an insightful narrative report.
Focus on: churn indicators, satisfaction scores, retention trends, customer segments, anomalies.
Structure: Executive Summary → Customer Health Overview → Churn/Retention Insights → Action Items.
Write for marketing and CX leadership."""
    user = f"""Dataset Schema:\n{state['df_summary']}\n\nAnalysis Results:\n{_build_tool_context(state)}
Write a complete customer analytics narrative report."""
    narrative = _llm_call(llm, system, user)
    return {**state, "agent_narrative": narrative}


def operations_agent_node(state: AnalyticsState, api_key: str) -> AnalyticsState:
    llm = get_llm(api_key)
    system = """You are an Operations Analytics Specialist.
Analyze operational metrics and write a clear narrative for operations managers.
Focus on: efficiency trends, downtime patterns, supply chain health, capacity utilization, anomalies.
Structure: Executive Summary → Operational KPIs → Bottlenecks & Anomalies → Optimization Recommendations.
Be specific about operational improvements."""
    user = f"""Dataset Schema:\n{state['df_summary']}\n\nAnalysis Results:\n{_build_tool_context(state)}
Write a complete operations analytics narrative report."""
    narrative = _llm_call(llm, system, user)
    return {**state, "agent_narrative": narrative}


# ─────────────────────────────────────────────
# Node: Output Guardrail + Final Insights LLM
# ─────────────────────────────────────────────
def output_guardrail_node(state: AnalyticsState, api_key: str) -> AnalyticsState:
    narrative = state.get("agent_narrative", "")
    is_valid, msg = validate_narrative(narrative)

    if not is_valid:
        # Fallback: regenerate with simpler prompt
        llm = get_llm(api_key, temperature=0.1)
        system = "You are a data analyst. Write a brief business report from the analysis data provided."
        user = f"Analysis data:\n{_build_tool_context(state)}\n\nWrite a concise narrative report."
        narrative = _llm_call(llm, system, user)

    # Final polish LLM pass
    llm = get_llm(api_key, temperature=0.2)
    system = """You are a professional business report editor.
Your task: Polish and finalize the analytics narrative.
- Add a clear title and date
- Ensure the report is well-structured with sections
- Highlight the TOP 3 key insights as bullet points at the top
- Add a concise "Next Steps" section at the end
- Keep total length between 400-800 words
- Use professional but accessible language"""

    user = f"""Draft narrative to polish:
{narrative}

Produce the final, polished business analytics report."""

    final = _llm_call(llm, system, user)
    final = sanitize_output(final)

    return {**state, "final_narrative": final}


# ─────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────
def build_graph(api_key: str):
    graph = StateGraph(AnalyticsState)

    # Add nodes
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("router", router_node)
    graph.add_node("run_mcp_tools", run_mcp_tools)
    graph.add_node("sales_agent", lambda s: sales_agent_node(s, api_key))
    graph.add_node("financial_agent", lambda s: financial_agent_node(s, api_key))
    graph.add_node("customer_agent", lambda s: customer_agent_node(s, api_key))
    graph.add_node("operations_agent", lambda s: operations_agent_node(s, api_key))
    graph.add_node("output_guardrail", lambda s: output_guardrail_node(s, api_key))

    # Edges
    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        lambda s: "end_error" if s.get("error") else "router",
        {"end_error": END, "router": "router"}
    )
    graph.add_edge("router", "run_mcp_tools")
    graph.add_conditional_edges(
        "run_mcp_tools",
        route_to_agent,
        {
            "sales_agent": "sales_agent",
            "financial_agent": "financial_agent",
            "customer_agent": "customer_agent",
            "operations_agent": "operations_agent",
        }
    )
    graph.add_edge("sales_agent", "output_guardrail")
    graph.add_edge("financial_agent", "output_guardrail")
    graph.add_edge("customer_agent", "output_guardrail")
    graph.add_edge("operations_agent", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame, api_key: str) -> dict:
    """
    Main entry point. Takes a DataFrame and returns the analysis state.
    """
    df_json = df.to_json(orient="records")
    graph = build_graph(api_key)

    initial_state: AnalyticsState = {
        "df_json": df_json,
        "df_summary": "",
        "dataset_type": "",
        "num_cols": [],
        "all_cols": [],
        "kpi_results": {},
        "trend_results": {},
        "anomaly_results": {},
        "forecast_results": {},
        "correlation_results": {},
        "agent_narrative": "",
        "final_narrative": "",
        "error": "",
    }

    result = graph.invoke(initial_state)
    return result
