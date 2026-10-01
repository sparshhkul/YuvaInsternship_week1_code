import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# YuvaIntern Week 1
# Relationship Analysis
# ============================================================

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CHARTS_DIR = BASE_DIR / "charts"

input_file = DATASET_DIR / "DataCo_Logistics_Cleaned.csv"

CHARTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("=" * 70)
print("YUVAINTERN WEEK 1 - RELATIONSHIP ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    input_file,
    encoding="latin1"
)

# ------------------------------------------------------------
# 3. Remove cancelled records
# ------------------------------------------------------------

valid_df = df[
    df["Delivery Status"]
    .astype(str)
    .str.lower()
    != "shipping canceled"
].copy()

# Official late flag
valid_df["Official Late"] = (
    valid_df["Delivery Status"]
    .astype(str)
    .str.lower()
    == "late delivery"
).astype(int)

print(f"\nTotal records: {len(df):,}")
print(f"Valid records: {len(valid_df):,}")

# ============================================================
# 4. NUMERICAL CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL CORRELATION ANALYSIS")
print("=" * 70)

# Variables relevant to logistics/business analysis
numeric_columns = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Delay",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Profit Per Order",
    "Benefit per order"
]

# Keep only columns that actually exist
numeric_columns = [
    column for column in numeric_columns
    if column in valid_df.columns
]

correlation_matrix = valid_df[numeric_columns].corr()

print(
    correlation_matrix.round(3)
)

# Save correlation matrix
correlation_matrix.to_csv(
    CHARTS_DIR / "correlation_matrix.csv"
)

# ============================================================
# 5. CORRELATION WITH DELIVERY DELAY
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH DELIVERY DELAY")
print("=" * 70)

delay_correlation = (
    correlation_matrix["Delivery Delay"]
    .drop("Delivery Delay")
    .sort_values(
        ascending=False
    )
)

print(
    delay_correlation.round(3)
)

delay_correlation.to_csv(
    CHARTS_DIR / "delay_correlations.csv"
)

# ============================================================
# CHART 1 - CORRELATION WITH DELIVERY DELAY
# ============================================================

print("\nCreating Chart 1...")

plt.figure(figsize=(10, 6))

plt.barh(
    delay_correlation.index,
    delay_correlation.values
)

plt.title(
    "Correlation of Variables with Delivery Delay",
    fontsize=16
)

plt.xlabel(
    "Correlation Coefficient",
    fontsize=12
)

plt.ylabel(
    "Variable",
    fontsize=12
)

plt.axvline(
    0,
    linewidth=1
)

plt.tight_layout()

chart1 = (
    CHARTS_DIR /
    "05_correlation_with_delivery_delay.png"
)

plt.savefig(
    chart1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart1}")

# ============================================================
# 6. DELAY BY ORDER ITEM QUANTITY
# ============================================================

print("\n" + "=" * 70)
print("DELIVERY DELAY BY ORDER ITEM QUANTITY")
print("=" * 70)

quantity_analysis = (
    valid_df
    .groupby("Order Item Quantity")
    .agg(
        Records=("Order Item Quantity", "size"),
        Average_Delay=("Delivery Delay", "mean"),
        Late_Delivery_Rate=("Official Late", "mean")
    )
    .sort_index()
)

quantity_analysis["Late_Delivery_Rate"] *= 100

print(
    quantity_analysis.round(2)
)

quantity_analysis.to_csv(
    CHARTS_DIR / "delay_by_quantity.csv"
)

# ============================================================
# CHART 2 - DELAY BY QUANTITY
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    quantity_analysis.index,
    quantity_analysis["Average_Delay"],
    marker="o"
)

plt.title(
    "Average Schedule Variance by Order Item Quantity",
    fontsize=16
)

plt.xlabel(
    "Order Item Quantity",
    fontsize=12
)

plt.ylabel(
    "Average Schedule Variance (Days)",
    fontsize=12
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

chart2 = (
    CHARTS_DIR /
    "06_delay_by_order_quantity.png"
)

plt.savefig(
    chart2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart2}")

# ============================================================
# 7. CUSTOMER SEGMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER SEGMENT ANALYSIS")
print("=" * 70)

segment_analysis = (
    valid_df
    .groupby("Customer Segment")
    .agg(
        Records=("Customer Segment", "size"),
        Average_Delay=("Delivery Delay", "mean"),
        Late_Delivery_Rate=("Official Late", "mean")
    )
    .sort_values(
        "Late_Delivery_Rate",
        ascending=False
    )
)

segment_analysis["Late_Delivery_Rate"] *= 100

print(
    segment_analysis.round(2)
)

segment_analysis.to_csv(
    CHARTS_DIR / "customer_segment_analysis.csv"
)

# ============================================================
# CHART 3 - CUSTOMER SEGMENT
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    segment_analysis.index,
    segment_analysis["Late_Delivery_Rate"]
)

plt.title(
    "Late Delivery Rate by Customer Segment",
    fontsize=16
)

plt.xlabel(
    "Customer Segment",
    fontsize=12
)

plt.ylabel(
    "Late Delivery Rate (%)",
    fontsize=12
)

plt.ylim(0, 100)

plt.tight_layout()

chart3 = (
    CHARTS_DIR /
    "07_late_delivery_by_customer_segment.png"
)

plt.savefig(
    chart3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart3}")

# ============================================================
# 8. MARKET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MARKET ANALYSIS")
print("=" * 70)

market_analysis = (
    valid_df
    .groupby("Market")
    .agg(
        Records=("Market", "size"),
        Average_Delay=("Delivery Delay", "mean"),
        Late_Delivery_Rate=("Official Late", "mean")
    )
    .sort_values(
        "Late_Delivery_Rate",
        ascending=False
    )
)

market_analysis["Late_Delivery_Rate"] *= 100

print(
    market_analysis.round(2)
)

market_analysis.to_csv(
    CHARTS_DIR / "market_analysis.csv"
)

# ============================================================
# CHART 4 - MARKET ANALYSIS
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    market_analysis.index,
    market_analysis["Late_Delivery_Rate"]
)

plt.title(
    "Late Delivery Rate by Market",
    fontsize=16
)

plt.xlabel(
    "Market",
    fontsize=12
)

plt.ylabel(
    "Late Delivery Rate (%)",
    fontsize=12
)

plt.ylim(0, 100)

plt.tight_layout()

chart4 = (
    CHARTS_DIR /
    "08_late_delivery_by_market.png"
)

plt.savefig(
    chart4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart4}")

# ============================================================
# 9. SALES AND DELAY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("SALES AND DELIVERY DELAY ANALYSIS")
print("=" * 70)

sales_groups = valid_df.copy()

# Create simple sales groups using quartiles
sales_groups["Sales_Group"] = pd.qcut(
    sales_groups["Sales"],
    q=4,
    labels=[
        "Low",
        "Medium-Low",
        "Medium-High",
        "High"
    ],
    duplicates="drop"
)

sales_analysis = (
    sales_groups
    .groupby("Sales_Group", observed=False)
    .agg(
        Records=("Sales", "size"),
        Average_Sales=("Sales", "mean"),
        Average_Delay=("Delivery Delay", "mean"),
        Late_Delivery_Rate=("Official Late", "mean")
    )
)

sales_analysis["Late_Delivery_Rate"] *= 100

print(
    sales_analysis.round(2)
)

sales_analysis.to_csv(
    CHARTS_DIR / "sales_group_analysis.csv"
)

# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("RELATIONSHIP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNew charts created:")

print("5. 05_correlation_with_delivery_delay.png")
print("6. 06_delay_by_order_quantity.png")
print("7. 07_late_delivery_by_customer_segment.png")
print("8. 08_late_delivery_by_market.png")

print("\nAnalysis tables saved inside:")
print(CHARTS_DIR)