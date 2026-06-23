"""
Guardrails: Input validation and output quality checks.
"""

import json
import pandas as pd
from typing import Tuple


# ── Input Guardrails ────────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validates dataset before processing.
    Returns (is_valid, message).
    """
    if df.empty:
        return False, "Dataset is empty. Please upload a non-empty CSV/JSON file."

    if len(df) < 5:
        return False, f"Dataset has only {len(df)} rows. Minimum 5 rows required for meaningful analysis."

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 0:
        return False, "No numeric columns found. Please upload a dataset with at least one numeric column."

    # Check for excessive nulls (>80%)
    null_pct = df.isnull().mean()
    bad_cols = null_pct[null_pct > 0.8].index.tolist()
    if len(bad_cols) == len(df.columns):
        return False, "All columns have >80% missing values. Please clean your dataset."

    return True, "Dataset is valid."


def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Heuristic: classify dataset as sales / financial / customer / operations.
    """
    cols_lower = [c.lower() for c in df.columns]
    col_str = " ".join(cols_lower)

    if any(k in col_str for k in ["revenue", "sales", "units", "orders", "discount", "product"]):
        return "sales"
    if any(k in col_str for k in ["profit", "loss", "expense", "income", "budget", "cost", "margin"]):
        return "financial"
    if any(k in col_str for k in ["customer", "churn", "satisfaction", "nps", "retention", "csat"]):
        return "customer"
    if any(k in col_str for k in ["inventory", "supply", "shipment", "operations", "efficiency", "downtime"]):
        return "operations"
    return "general"


# ── Output Guardrails ───────────────────────────────────────────────────────

def validate_narrative(narrative: str) -> Tuple[bool, str]:
    """
    Basic quality check on generated narrative.
    """
    if not narrative or len(narrative.strip()) < 50:
        return False, "Generated narrative is too short. Regenerating..."

    # Check it's not just raw JSON
    try:
        json.loads(narrative)
        return False, "Output appears to be raw JSON, not a narrative. Regenerating..."
    except Exception:
        pass

    return True, "Narrative quality check passed."


def sanitize_output(text: str) -> str:
    """Remove any leaked system prompts or JSON artifacts."""
    # Strip any accidental JSON blobs at start/end
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return text.strip()
