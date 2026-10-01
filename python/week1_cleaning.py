import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

input_file = DATASET_DIR / "DataCoSupplyChainDataset.csv"

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(input_file, encoding="latin1")

print("Original dataset shape:")
print(df.shape)

# --------------------------------------------------
# 3. Remove unnecessary / sensitive columns
# --------------------------------------------------

columns_to_remove = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Order Zipcode",
    "Product Description",
    "Product Image",
    "Product Status"
]

df_clean = df.drop(columns=columns_to_remove, errors="ignore")

# --------------------------------------------------
# 4. Create delay variable
# --------------------------------------------------

df_clean["Delivery Delay"] = (
    df_clean["Days for shipping (real)"]
    - df_clean["Days for shipment (scheduled)"]
)

# --------------------------------------------------
# 5. Create delayed flag
# --------------------------------------------------

df_clean["Is Delayed"] = (
    df_clean["Delivery Delay"] > 0
).astype(int)

# --------------------------------------------------
# 6. Convert dates
# --------------------------------------------------

df_clean["order date (DateOrders)"] = pd.to_datetime(
    df_clean["order date (DateOrders)"],
    errors="coerce"
)

df_clean["shipping date (DateOrders)"] = pd.to_datetime(
    df_clean["shipping date (DateOrders)"],
    errors="coerce"
)

# --------------------------------------------------
# 7. Show cleaned dataset information
# --------------------------------------------------

print("\nCleaned dataset shape:")
print(df_clean.shape)

print("\nRemaining missing values:")
print(df_clean.isnull().sum().sort_values(ascending=False).head(15))

print("\nDelay distribution:")
print(df_clean["Delivery Delay"].describe())

print("\nDelayed deliveries:")
print(df_clean["Is Delayed"].value_counts())

# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

output_file = DATASET_DIR / "DataCo_Logistics_Cleaned.csv"

df_clean.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved to:")
print(output_file)