"""
MCP Server for Data Analytics Tools
Provides: Trend Analyzer, Anomaly Detector, Forecast Analyzer,
          KPI Summarizer, Correlation Analyzer
"""

import json
import numpy as np
import pandas as pd
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


app = Server("analytics-tools")


def _df_from_json(data_json: str) -> pd.DataFrame:
    data = json.loads(data_json)
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    raise ValueError("data_json must be a JSON array or object")


# ─────────────────────────────────────────────
# TOOL 1: Trend Analyzer
# ─────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="trend_analyzer",
            description=(
                "Detects trends in numeric columns of a dataset. "
                "Returns direction (upward/downward/stable), slope, and trend summary per column."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_json": {
                        "type": "string",
                        "description": "JSON string of the dataset (list of records or column-oriented dict)"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Numeric columns to analyze (optional; all numeric cols if omitted)"
                    }
                },
                "required": ["data_json"]
            }
        ),
        types.Tool(
            name="anomaly_detector",
            description=(
                "Detects anomalies/outliers in numeric columns using Z-score and IQR methods. "
                "Returns anomaly counts, affected rows, and severity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_json": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "z_threshold": {
                        "type": "number",
                        "description": "Z-score threshold for anomaly detection (default 2.5)"
                    }
                },
                "required": ["data_json"]
            }
        ),
        types.Tool(
            name="forecast_analyzer",
            description=(
                "Generates simple moving-average and linear forecast for numeric time-series columns. "
                "Returns next-period forecasts and confidence info."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_json": {"type": "string"},
                    "target_column": {"type": "string", "description": "Column to forecast"},
                    "periods_ahead": {"type": "integer", "description": "Number of future periods (default 3)"}
                },
                "required": ["data_json", "target_column"]
            }
        ),
        types.Tool(
            name="kpi_summarizer",
            description=(
                "Computes key performance indicators: mean, median, std dev, min, max, "
                "growth rate, and percentile distribution for numeric columns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_json": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["data_json"]
            }
        ),
        types.Tool(
            name="correlation_analyzer",
            description=(
                "Computes pairwise Pearson correlation between numeric columns and identifies "
                "strong positive/negative relationships."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_json": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "threshold": {
                        "type": "number",
                        "description": "Absolute correlation threshold to flag (default 0.7)"
                    }
                },
                "required": ["data_json"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:

    # ── Trend Analyzer ──────────────────────────────────────
    if name == "trend_analyzer":
        df = _df_from_json(arguments["data_json"])
        cols = arguments.get("columns") or df.select_dtypes(include="number").columns.tolist()
        results = {}
        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna().reset_index(drop=True)
            if len(series) < 2:
                continue
            x = np.arange(len(series))
            slope, intercept = np.polyfit(x, series, 1)
            pct_change = ((series.iloc[-1] - series.iloc[0]) / (abs(series.iloc[0]) + 1e-9)) * 100
            direction = "upward" if slope > 0.01 * series.mean() else (
                "downward" if slope < -0.01 * series.mean() else "stable"
            )
            results[col] = {
                "direction": direction,
                "slope": round(float(slope), 4),
                "pct_change_overall": round(float(pct_change), 2),
                "start_value": round(float(series.iloc[0]), 2),
                "end_value": round(float(series.iloc[-1]), 2),
                "summary": (
                    f"{col} shows a {direction} trend with {abs(pct_change):.1f}% "
                    f"{'increase' if pct_change >= 0 else 'decrease'} over the period."
                )
            }
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

    # ── Anomaly Detector ────────────────────────────────────
    elif name == "anomaly_detector":
        df = _df_from_json(arguments["data_json"])
        cols = arguments.get("columns") or df.select_dtypes(include="number").columns.tolist()
        z_thresh = arguments.get("z_threshold", 2.5)
        results = {}
        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            mean, std = series.mean(), series.std()
            if std == 0:
                continue
            z_scores = np.abs((series - mean) / std)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            iqr_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            z_mask = z_scores > z_thresh
            anomaly_indices = series[z_mask | iqr_mask].index.tolist()
            results[col] = {
                "anomaly_count": len(anomaly_indices),
                "anomaly_percentage": round(len(anomaly_indices) / len(series) * 100, 2),
                "anomaly_row_indices": anomaly_indices[:10],
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
                "severity": "high" if len(anomaly_indices) / len(series) > 0.1 else (
                    "medium" if len(anomaly_indices) > 0 else "none"
                ),
                "summary": (
                    f"{col} has {len(anomaly_indices)} anomalies "
                    f"({len(anomaly_indices)/len(series)*100:.1f}% of data)."
                )
            }
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

    # ── Forecast Analyzer ───────────────────────────────────
    elif name == "forecast_analyzer":
        df = _df_from_json(arguments["data_json"])
        col = arguments["target_column"]
        periods = arguments.get("periods_ahead", 3)
        series = df[col].dropna().reset_index(drop=True)
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        forecasts = []
        for i in range(1, periods + 1):
            forecasts.append(round(float(intercept + slope * (len(series) + i - 1)), 2))
        window = min(5, len(series))
        ma_base = float(series.iloc[-window:].mean())
        ma_forecasts = [round(ma_base + slope * i, 2) for i in range(1, periods + 1)]
        result = {
            "target_column": col,
            "linear_forecasts": forecasts,
            "moving_avg_forecasts": ma_forecasts,
            "last_known_value": round(float(series.iloc[-1]), 2),
            "trend_slope": round(float(slope), 4),
            "summary": (
                f"Forecast for {col}: next {periods} period(s) projected at "
                f"{forecasts} (linear trend). "
                f"{'Upward' if slope > 0 else 'Downward'} momentum expected."
            )
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── KPI Summarizer ──────────────────────────────────────
    elif name == "kpi_summarizer":
        df = _df_from_json(arguments["data_json"])
        cols = arguments.get("columns") or df.select_dtypes(include="number").columns.tolist()
        results = {}
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            growth = ((s.iloc[-1] - s.iloc[0]) / (abs(s.iloc[0]) + 1e-9) * 100) if len(s) > 1 else 0
            results[col] = {
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std_dev": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "growth_rate_pct": round(float(growth), 2),
                "p25": round(float(s.quantile(0.25)), 2),
                "p75": round(float(s.quantile(0.75)), 2),
                "count": int(s.count()),
                "summary": (
                    f"{col}: avg={s.mean():.2f}, range=[{s.min():.2f}–{s.max():.2f}], "
                    f"growth={growth:.1f}%"
                )
            }
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

    # ── Correlation Analyzer ────────────────────────────────
    elif name == "correlation_analyzer":
        df = _df_from_json(arguments["data_json"])
        cols = arguments.get("columns") or df.select_dtypes(include="number").columns.tolist()
        threshold = arguments.get("threshold", 0.7)
        num_df = df[cols].select_dtypes(include="number")
        corr_matrix = num_df.corr().round(3)
        strong_pairs = []
        for i, c1 in enumerate(corr_matrix.columns):
            for j, c2 in enumerate(corr_matrix.columns):
                if i >= j:
                    continue
                val = corr_matrix.loc[c1, c2]
                if abs(val) >= threshold:
                    strong_pairs.append({
                        "col_a": c1,
                        "col_b": c2,
                        "correlation": round(float(val), 3),
                        "relationship": "strong positive" if val > 0 else "strong negative"
                    })
        result = {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_pairs,
            "summary": (
                f"Found {len(strong_pairs)} strong correlation(s) (|r|>={threshold}). "
                + (
                    "; ".join(
                        f"{p['col_a']} & {p['col_b']} ({p['correlation']})"
                        for p in strong_pairs
                    ) if strong_pairs else "No strong correlations detected."
                )
            )
        }
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
