"""Generate sample datasets for demo purposes."""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

def generate_sales_data():
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return pd.DataFrame({
        "month": months,
        "revenue": [120000,135000,128000,142000,138000,155000,162000,149000,171000,185000,178000,210000],
        "units_sold": [1200,1350,1280,1420,1380,1550,1620,1490,1710,1850,1780,2100],
        "discount_pct": [5,5,8,6,7,5,4,6,5,4,3,6],
        "new_customers": [45,52,48,60,55,70,72,65,80,90,85,110],
        "orders": [980,1100,1050,1180,1150,1300,1350,1250,1450,1550,1500,1800],
        "avg_order_value": [122,123,122,120,120,119,120,119,118,119,119,117],
        "returns": [45,52,60,48,55,50,42,58,49,40,38,65]
    })

def generate_financial_data():
    quarters = ["Q1-2023","Q2-2023","Q3-2023","Q4-2023","Q1-2024","Q2-2024","Q3-2024","Q4-2024"]
    return pd.DataFrame({
        "quarter": quarters,
        "revenue": [500000,520000,510000,580000,560000,600000,590000,650000],
        "expenses": [380000,390000,385000,420000,400000,430000,415000,460000],
        "profit": [120000,130000,125000,160000,160000,170000,175000,190000],
        "profit_margin": [24,25,24.5,27.6,28.6,28.3,29.7,29.2],
        "operating_cost": [200000,210000,205000,230000,220000,235000,225000,250000],
        "capex": [50000,55000,48000,62000,58000,65000,60000,70000],
        "ebitda": [150000,160000,155000,190000,190000,200000,205000,220000]
    })

def generate_customer_data():
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return pd.DataFrame({
        "month": months,
        "total_customers": [5000,5200,5150,5400,5350,5600,5700,5650,5900,6100,6050,6300],
        "churned_customers": [120,110,150,100,130,90,85,95,80,75,70,65],
        "nps_score": [42,44,41,46,45,48,50,49,52,54,53,56],
        "csat_score": [3.8,3.9,3.7,4.0,3.9,4.1,4.2,4.1,4.3,4.4,4.3,4.5],
        "avg_lifetime_value": [850,860,855,875,870,890,900,895,915,930,925,950],
        "support_tickets": [320,310,340,290,305,280,260,270,250,240,235,220],
        "retention_rate": [97.6,97.9,97.1,98.1,97.6,98.4,98.5,98.3,98.6,98.8,98.8,99.0]
    })

def generate_operations_data():
    weeks = [f"W{i}" for i in range(1, 13)]
    return pd.DataFrame({
        "week": weeks,
        "production_units": [2400,2350,2500,2450,2600,2550,2700,2650,2800,2750,2900,2850],
        "efficiency_pct": [82,80,84,83,86,85,87,86,88,87,89,88],
        "downtime_hours": [12,15,10,11,8,9,7,8,6,7,5,6],
        "defect_rate": [2.1,2.3,1.9,2.0,1.7,1.8,1.5,1.6,1.4,1.5,1.2,1.3],
        "inventory_turnover": [8.2,8.0,8.5,8.3,8.8,8.6,9.0,8.9,9.2,9.1,9.5,9.3],
        "on_time_delivery_pct": [91,89,92,91,93,92,94,93,95,94,96,95],
        "cost_per_unit": [42,43,41,42,40,41,39,40,38,39,37,38]
    })


SAMPLE_DATASETS = {
    "Sales Performance": generate_sales_data,
    "Financial Overview": generate_financial_data,
    "Customer Analytics": generate_customer_data,
    "Operations Metrics": generate_operations_data,
}

def get_sample_dataset(name: str) -> pd.DataFrame:
    return SAMPLE_DATASETS[name]()
