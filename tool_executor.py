"""
Local MCP Tool Executor
Runs analytics tools in-process (no subprocess needed for Streamlit demo).
Mirrors the MCP server tools but callable directly from agents.
"""

import json
import numpy as np
import pandas as pd
from typing import Any


def _df_from_json(data_json: str) -> pd.DataFrame:
    data = json.loads(data_json)
    return pd.DataFrame(data)


def trend_analyzer(data_json: str, columns: list[str] | None = None) -> dict:
    df = _df_from_json(data_json)
    cols = columns or df.select_dtypes(include="number").columns.tolist()
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
    return results


def anomaly_detector(data_json: str, columns: list[str] | None = None, z_threshold: float = 2.5) -> dict:
    df = _df_from_json(data_json)
    cols = columns or df.select_dtypes(include="number").columns.tolist()
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
        z_mask = z_scores > z_threshold
        anomaly_indices = series[z_mask | iqr_mask].index.tolist()
        results[col] = {
            "anomaly_count": len(anomaly_indices),
            "anomaly_percentage": round(len(anomaly_indices) / len(series) * 100, 2),
            "anomaly_row_indices": anomaly_indices[:10],
            "severity": "high" if len(anomaly_indices) / len(series) > 0.1 else (
                "medium" if len(anomaly_indices) > 0 else "none"
            ),
            "summary": (
                f"{col} has {len(anomaly_indices)} anomalies "
                f"({len(anomaly_indices)/len(series)*100:.1f}% of data)."
            )
        }
    return results


def forecast_analyzer(data_json: str, target_column: str, periods_ahead: int = 3) -> dict:
    df = _df_from_json(data_json)
    col = target_column
    series = df[col].dropna().reset_index(drop=True)
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    forecasts = [round(float(intercept + slope * (len(series) + i - 1)), 2) for i in range(1, periods_ahead + 1)]
    window = min(5, len(series))
    ma_base = float(series.iloc[-window:].mean())
    ma_forecasts = [round(ma_base + slope * i, 2) for i in range(1, periods_ahead + 1)]
    return {
        "target_column": col,
        "linear_forecasts": forecasts,
        "moving_avg_forecasts": ma_forecasts,
        "last_known_value": round(float(series.iloc[-1]), 2),
        "trend_slope": round(float(slope), 4),
        "summary": (
            f"Forecast for {col}: next {periods_ahead} period(s) projected at {forecasts}. "
            f"{'Upward' if slope > 0 else 'Downward'} momentum expected."
        )
    }


def kpi_summarizer(data_json: str, columns: list[str] | None = None) -> dict:
    df = _df_from_json(data_json)
    cols = columns or df.select_dtypes(include="number").columns.tolist()
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
                f"{col}: avg={s.mean():.2f}, range=[{s.min():.2f}–{s.max():.2f}], growth={growth:.1f}%"
            )
        }
    return results


def correlation_analyzer(data_json: str, columns: list[str] | None = None, threshold: float = 0.7) -> dict:
    df = _df_from_json(data_json)
    cols = columns or df.select_dtypes(include="number").columns.tolist()
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
                    "col_a": c1, "col_b": c2,
                    "correlation": round(float(val), 3),
                    "relationship": "strong positive" if val > 0 else "strong negative"
                })
    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "strong_correlations": strong_pairs,
        "summary": (
            f"Found {len(strong_pairs)} strong correlation(s) (|r|>={threshold}). "
            + ("; ".join(f"{p['col_a']} & {p['col_b']} ({p['correlation']})" for p in strong_pairs)
               if strong_pairs else "No strong correlations detected.")
        )
    }


TOOL_REGISTRY: dict[str, Any] = {
    "trend_analyzer": trend_analyzer,
    "anomaly_detector": anomaly_detector,
    "forecast_analyzer": forecast_analyzer,
    "kpi_summarizer": kpi_summarizer,
    "correlation_analyzer": correlation_analyzer,
}
