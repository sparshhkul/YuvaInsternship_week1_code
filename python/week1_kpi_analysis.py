import pandas as pd
from pathlib import Path

# ============================================================
# YuvaIntern Week 1
# Logistics Data Analysis - KPI Analysis
# ============================================================

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CHARTS_DIR = BASE_DIR / "charts"

input_file = DATASET_DIR / "DataCo_Logistics_Cleaned.csv"

# Create charts/output folder if it does not exist
CHARTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. Load cleaned dataset
# ------------------------------------------------------------

print("=" * 70)
print("YUVAINTERN WEEK 1 - KPI ANALYSIS")
print("=" * 70)

print("\nLoading cleaned dataset...")

try:
    df = pd.read_csv(input_file, encoding="latin1")
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error while loading dataset:")
    print(e)
    raise

print(f"\nTotal records: {len(df):,}")

# ------------------------------------------------------------
# 3. Create valid delivery dataset
# ------------------------------------------------------------
# Cancelled shipments are excluded from delivery-performance KPIs.

valid_df = df[
    df["Delivery Status"]
    .astype(str)
    .str.lower()
    != "shipping canceled"
].copy()

print(f"Valid non-cancelled records: {len(valid_df):,}")

# ------------------------------------------------------------
# 4. Create official late-delivery flag
# ------------------------------------------------------------

valid_df["Official Late"] = (
    valid_df["Delivery Status"]
    .astype(str)
    .str.lower()
    == "late delivery"
).astype(int)

# ------------------------------------------------------------
# 5. DELIVERY STATUS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DELIVERY STATUS DISTRIBUTION")
print("=" * 70)

status_counts = df["Delivery Status"].value_counts()

print(status_counts)

# Save status distribution
status_counts.to_csv(
    CHARTS_DIR / "delivery_status_distribution.csv"
)

# ------------------------------------------------------------
# 6. KPI 1 - LATE DELIVERY RATE
# ------------------------------------------------------------

late_count = valid_df["Official Late"].sum()

valid_delivery_records = len(valid_df)

late_delivery_rate = (
    late_count / valid_delivery_records * 100
)

print("\n" + "=" * 70)
print("KPI 1 - LATE DELIVERY RATE")
print("=" * 70)

print(f"Late delivery records : {late_count:,}")
print(f"Valid delivery records: {valid_delivery_records:,}")
print(f"Late delivery rate    : {late_delivery_rate:.2f}%")

# ------------------------------------------------------------
# 7. KPI 2 - AVERAGE ACTUAL SHIPPING TIME
# ------------------------------------------------------------

average_actual_shipping = (
    valid_df["Days for shipping (real)"].mean()
)

print("\n" + "=" * 70)
print("KPI 2 - AVERAGE ACTUAL SHIPPING TIME")
print("=" * 70)

print(
    f"Average actual shipping time: "
    f"{average_actual_shipping:.2f} days"
)

# ------------------------------------------------------------
# 8. KPI 3 - AVERAGE SCHEDULED SHIPPING TIME
# ------------------------------------------------------------

average_scheduled_shipping = (
    valid_df["Days for shipment (scheduled)"].mean()
)

print("\n" + "=" * 70)
print("KPI 3 - AVERAGE SCHEDULED SHIPPING TIME")
print("=" * 70)

print(
    f"Average scheduled shipping time: "
    f"{average_scheduled_shipping:.2f} days"
)

# ------------------------------------------------------------
# 9. KPI 4 - AVERAGE SCHEDULE VARIANCE
# ------------------------------------------------------------
# Schedule variance:
# Actual shipping days - Scheduled shipping days
#
# Positive value  = actual time exceeded schedule
# Negative value  = shipment was completed earlier
# Zero            = actual and scheduled time were equal

average_schedule_variance = (
    valid_df["Delivery Delay"].mean()
)

print("\n" + "=" * 70)
print("KPI 4 - AVERAGE SCHEDULE VARIANCE")
print("=" * 70)

print(
    f"Average schedule variance: "
    f"{average_schedule_variance:.2f} days"
)

# ------------------------------------------------------------
# 10. ADDITIONAL KPI - AVERAGE DELAY AMONG LATE DELIVERIES
# ------------------------------------------------------------

late_only = valid_df[
    valid_df["Official Late"] == 1
].copy()

average_late_delay = (
    late_only["Delivery Delay"].mean()
)

median_late_delay = (
    late_only["Delivery Delay"].median()
)

print("\n" + "=" * 70)
print("ADDITIONAL KPI - DELAY AMONG LATE DELIVERIES")
print("=" * 70)

print(
    f"Late delivery records      : "
    f"{len(late_only):,}"
)

print(
    f"Average delay when late    : "
    f"{average_late_delay:.2f} days"
)

print(
    f"Median delay when late     : "
    f"{median_late_delay:.2f} days"
)

# ------------------------------------------------------------
# 11. SHIPPING MODE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SHIPPING MODE ANALYSIS")
print("=" * 70)

shipping_analysis = (
    valid_df
    .groupby("Shipping Mode")
    .agg(
        Records=("Shipping Mode", "size"),

        Average_Actual_Days=(
            "Days for shipping (real)",
            "mean"
        ),

        Average_Scheduled_Days=(
            "Days for shipment (scheduled)",
            "mean"
        ),

        Average_Schedule_Variance=(
            "Delivery Delay",
            "mean"
        ),

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

# Convert proportion to percentage
shipping_analysis["Late_Delivery_Rate"] *= 100

print(
    shipping_analysis.round(2)
)

# Save shipping-mode analysis
shipping_analysis.to_csv(
    CHARTS_DIR / "shipping_mode_analysis.csv"
)

# ------------------------------------------------------------
# 12. DELIVERY STATUS BY SHIPPING MODE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DELIVERY STATUS BY SHIPPING MODE")
print("=" * 70)

status_by_mode = pd.crosstab(
    valid_df["Shipping Mode"],
    valid_df["Delivery Status"]
)

print(status_by_mode)

# Save table
status_by_mode.to_csv(
    CHARTS_DIR / "delivery_status_by_shipping_mode.csv"
)

# ------------------------------------------------------------
# 13. AVERAGE SHIPPING DAYS BY SHIPPING MODE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AVERAGE SHIPPING DAYS BY SHIPPING MODE")
print("=" * 70)

mode_days = (
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
        ),

        Average_Schedule_Variance=(
            "Delivery Delay",
            "mean"
        )
    )
)

print(
    mode_days.round(2)
)

# Save table
mode_days.to_csv(
    CHARTS_DIR / "shipping_mode_days.csv"
)

# ------------------------------------------------------------
# 14. REGION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REGION ANALYSIS")
print("=" * 70)

region_analysis = (
    valid_df
    .groupby("Order Region")
    .agg(
        Records=("Order Region", "size"),

        Average_Schedule_Variance=(
            "Delivery Delay",
            "mean"
        ),

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

# Convert proportion to percentage
region_analysis["Late_Delivery_Rate"] *= 100

print(
    region_analysis.round(2)
)

# Save regional analysis
region_analysis.to_csv(
    CHARTS_DIR / "region_analysis.csv"
)

# ------------------------------------------------------------
# 15. ORDER / RECORD VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ORDER / RECORD VALIDATION")
print("=" * 70)

unique_orders = df["Order Id"].nunique()

unique_order_items = (
    df["Order Item Id"].nunique()
)

unique_customers = (
    df["Order Customer Id"].nunique()
)

print(f"Total dataset records      : {len(df):,}")
print(f"Unique Order IDs           : {unique_orders:,}")
print(f"Unique Order Item IDs      : {unique_order_items:,}")
print(f"Unique Customers           : {unique_customers:,}")

# Records associated with each Order ID
order_record_counts = (
    df.groupby("Order Id").size()
)

print(
    f"\nAverage records per Order ID: "
    f"{order_record_counts.mean():.2f}"
)

print(
    f"Maximum records for one Order ID: "
    f"{order_record_counts.max()}"
)

# ------------------------------------------------------------
# 16. LATE DELIVERY ANALYSIS BY SHIPPING MODE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LATE DELIVERY ANALYSIS BY SHIPPING MODE")
print("=" * 70)

late_mode_analysis = (
    late_only
    .groupby("Shipping Mode")
    .agg(
        Late_Records=("Shipping Mode", "size"),

        Average_Late_Delay=(
            "Delivery Delay",
            "mean"
        ),

        Median_Late_Delay=(
            "Delivery Delay",
            "median"
        )
    )
    .sort_values(
        "Average_Late_Delay",
        ascending=False
    )
)

print(
    late_mode_analysis.round(2)
)

# Save result
late_mode_analysis.to_csv(
    CHARTS_DIR / "late_delivery_by_shipping_mode.csv"
)

# ------------------------------------------------------------
# 17. KPI SUMMARY TABLE
# ------------------------------------------------------------

kpi_summary = pd.DataFrame({
    "KPI": [
        "Late Delivery Rate",
        "Average Actual Shipping Time",
        "Average Scheduled Shipping Time",
        "Average Schedule Variance",
        "Average Delay Among Late Deliveries",
        "Median Delay Among Late Deliveries"
    ],

    "Value": [
        late_delivery_rate,
        average_actual_shipping,
        average_scheduled_shipping,
        average_schedule_variance,
        average_late_delay,
        median_late_delay
    ],

    "Unit": [
        "%",
        "days",
        "days",
        "days",
        "days",
        "days"
    ]
})

print("\n" + "=" * 70)
print("FINAL KPI SUMMARY")
print("=" * 70)

print(kpi_summary.to_string(index=False))

# Save KPI summary
kpi_summary.to_csv(
    CHARTS_DIR / "kpi_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 18. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("KPI ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput files saved in:")
print(CHARTS_DIR)

print("\nGenerated files:")
print("1. delivery_status_distribution.csv")
print("2. shipping_mode_analysis.csv")
print("3. delivery_status_by_shipping_mode.csv")
print("4. shipping_mode_days.csv")
print("5. region_analysis.csv")
print("6. late_delivery_by_shipping_mode.csv")
print("7. kpi_summary.csv")