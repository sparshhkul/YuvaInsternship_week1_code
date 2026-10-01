import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define file locations
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

main_file = DATASET_DIR / "DataCoSupplyChainDataset.csv"
description_file = DATASET_DIR / "DescriptionDataCoSupplyChain.csv"

print("=" * 60)
print("YUVAINTERN WEEK 1 - DATASET INSPECTION")
print("=" * 60)

# --------------------------------------------------
# 2. Check whether files exist
# --------------------------------------------------

print("\nChecking files...")

if main_file.exists():
    print(f"Main dataset found: {main_file.name}")
else:
    print("ERROR: Main dataset not found.")

if description_file.exists():
    print(f"Description file found: {description_file.name}")
else:
    print("ERROR: Description file not found.")

# --------------------------------------------------
# 3. Load the main dataset
# --------------------------------------------------

print("\nLoading main dataset...")

try:
    df = pd.read_csv(main_file, encoding="latin1")
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error while loading dataset:")
    print(e)
    raise

# --------------------------------------------------
# 4. Basic information
# --------------------------------------------------

print("\n" + "=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

# --------------------------------------------------
# 5. Column names
# --------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# --------------------------------------------------
# 6. First five records
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())

# --------------------------------------------------
# 7. Data types
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# --------------------------------------------------
# 8. Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

# --------------------------------------------------
# 9. Duplicate records
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicates}")

# --------------------------------------------------
# 10. Statistical summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe())

# --------------------------------------------------
# 11. Save a small summary
# --------------------------------------------------

summary_file = BASE_DIR / "python" / "dataset_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("YUVAINTERN WEEK 1 DATASET SUMMARY\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n\n")

    file.write("COLUMN NAMES\n")
    file.write("-" * 50 + "\n")

    for column in df.columns:
        file.write(f"{column}\n")

    file.write("\nMISSING VALUES\n")
    file.write("-" * 50 + "\n")
    file.write(str(missing_values))

    file.write("\n\nDUPLICATE ROWS\n")
    file.write("-" * 50 + "\n")
    file.write(str(duplicates))

print("\n" + "=" * 60)
print("Inspection complete!")
print(f"Summary saved to: {summary_file}")
print("=" * 60)