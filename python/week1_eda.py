import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# YuvaIntern Week 1
# Exploratory Data Analysis (EDA)
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
print("YUVAINTERN WEEK 1 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nLoading cleaned dataset...")

df = pd.read_csv(
    input_file,
    encoding="latin1"
)

print("Dataset loaded successfully.")

print(f"Total records: {len(df):,}")

# ------------------------------------------------------------
# 3. Remove cancelled records for delivery analysis
# ------------------------------------------------------------

valid_df = df[
    df["Delivery Status"]
    .astype(str)
    .str.lower()
    != "shipping canceled"
].copy()

# Official late-delivery flag
valid_df["Official Late"] = (
    valid_df["Delivery Status"]
    .astype(str)
    .str.lower()
    == "late delivery"
).astype(int)

print(
    f"Valid non-cancelled records: "
    f"{len(valid_df):,}"
)

# ============================================================
# CHART 1
# DELIVERY STATUS DISTRIBUTION
# ============================================================

print("\nCreating Chart 1...")

status_counts = df["Delivery Status"].value_counts()

plt.figure(figsize=(10, 6))

plt.bar(
    status_counts.index,
    status_counts.values
)

plt.title(
    "Delivery Status Distribution",
    fontsize=16
)

plt.xlabel(
    "Delivery Status",
    fontsize=12
)

plt.ylabel(
    "Number of Records",
    fontsize=12
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

chart1 = CHARTS_DIR / "01_delivery_status_distribution.png"

plt.savefig(
    chart1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart1}")

# ============================================================
# CHART 2
# LATE DELIVERY RATE BY SHIPPING MODE
# ============================================================

print("\nCreating Chart 2...")

shipping_mode_analysis = (
    valid_df
    .groupby("Shipping Mode")
    .agg(
        Late_Delivery_Rate=(
            "Official Late",
            "mean"
        )
    )
    .sort_values(
        "Late_Delivery_Rate",
        ascending=False
    )
)

shipping_mode_analysis[
    "Late_Delivery_Rate"
] *= 100

plt.figure(figsize=(9, 6))

plt.bar(
    shipping_mode_analysis.index,
    shipping_mode_analysis["Late_Delivery_Rate"]
)

plt.title(
    "Late Delivery Rate by Shipping Mode",
    fontsize=16
)

plt.xlabel(
    "Shipping Mode",
    fontsize=12
)

plt.ylabel(
    "Late Delivery Rate (%)",
    fontsize=12
)

plt.ylim(0, 110)

# Display percentages above bars
for i, value in enumerate(
    shipping_mode_analysis["Late_Delivery_Rate"]
):
    plt.text(
        i,
        value + 2,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()

chart2 = CHARTS_DIR / "02_late_delivery_rate_by_shipping_mode.png"

plt.savefig(
    chart2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart2}")

# ============================================================
# CHART 3
# ACTUAL VS SCHEDULED SHIPPING DAYS
# ============================================================

print("\nCreating Chart 3...")

shipping_days = (
    valid_df
    .groupby("Shipping Mode")
    .agg(
        Actual_Days=(
            "Days for shipping (real)",
            "mean"
        ),

        Scheduled_Days=(
            "Days for shipment (scheduled)",
            "mean"
        )
    )
)

shipping_days = shipping_days.sort_index()

x = range(len(shipping_days))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    shipping_days["Actual_Days"],
    width=width,
    label="Actual Shipping Days"
)

plt.bar(
    [i + width / 2 for i in x],
    shipping_days["Scheduled_Days"],
    width=width,
    label="Scheduled Shipping Days"
)

plt.title(
    "Actual vs Scheduled Shipping Days by Shipping Mode",
    fontsize=16
)

plt.xlabel(
    "Shipping Mode",
    fontsize=12
)

plt.ylabel(
    "Average Days",
    fontsize=12
)

plt.xticks(
    list(x),
    shipping_days.index
)

plt.legend()

plt.tight_layout()

chart3 = CHARTS_DIR / "03_actual_vs_scheduled_shipping_days.png"

plt.savefig(
    chart3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart3}")

# ============================================================
# CHART 4
# LATE DELIVERY RATE BY REGION
# ============================================================

print("\nCreating Chart 4...")

region_analysis = (
    valid_df
    .groupby("Order Region")
    .agg(
        Late_Delivery_Rate=(
            "Official Late",
            "mean"
        )
    )
    .sort_values(
        "Late_Delivery_Rate",
        ascending=True
    )
)

region_analysis[
    "Late_Delivery_Rate"
] *= 100

plt.figure(figsize=(11, 10))

plt.barh(
    region_analysis.index,
    region_analysis["Late_Delivery_Rate"]
)

plt.title(
    "Late Delivery Rate by Region",
    fontsize=16
)

plt.xlabel(
    "Late Delivery Rate (%)",
    fontsize=12
)

plt.ylabel(
    "Order Region",
    fontsize=12
)

plt.xlim(0, 70)

plt.tight_layout()

chart4 = CHARTS_DIR / "04_late_delivery_rate_by_region.png"

plt.savefig(
    chart4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {chart4}")

# ============================================================
# 5. Save EDA tables
# ============================================================

shipping_mode_analysis.to_csv(
    CHARTS_DIR / "eda_shipping_mode_late_rate.csv"
)

shipping_days.to_csv(
    CHARTS_DIR / "eda_shipping_mode_shipping_days.csv"
)

region_analysis.to_csv(
    CHARTS_DIR / "eda_region_late_rate.csv"
)

# ============================================================
# 6. Final message
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCharts created:")

print("1. 01_delivery_status_distribution.png")
print("2. 02_late_delivery_rate_by_shipping_mode.png")
print("3. 03_actual_vs_scheduled_shipping_days.png")
print("4. 04_late_delivery_rate_by_region.png")

print("\nAll charts are saved inside:")
print(CHARTS_DIR)