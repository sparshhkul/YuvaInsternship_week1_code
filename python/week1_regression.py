import pandas as pd
from pathlib import Path

from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# YuvaIntern Week 1
# Regression Analysis - Shipping Time Prediction
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
print("YUVAINTERN WEEK 1 - REGRESSION ANALYSIS")
print("=" * 70)

print("\nLoading cleaned dataset...")

try:
    df = pd.read_csv(
        input_file,
        encoding="latin1"
    )

    print("Dataset loaded successfully.")

except Exception as e:
    print("\nError while loading dataset:")
    print(e)
    raise

# ------------------------------------------------------------
# 3. Remove cancelled records
# ------------------------------------------------------------

df = df[
    df["Delivery Status"]
    .astype(str)
    .str.lower()
    != "shipping canceled"
].copy()

print(
    f"\nRecords used for modelling: "
    f"{len(df):,}"
)

# ------------------------------------------------------------
# 4. Target variable
# ------------------------------------------------------------
# We are predicting actual shipping duration.

target = "Days for shipping (real)"

# ------------------------------------------------------------
# 5. Predictor variables
# ------------------------------------------------------------
# These are operational/business variables that may be
# available for planning or analysis.

features = [
    "Shipping Mode",
    "Order Region",
    "Market",
    "Customer Segment",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio"
]

# Keep only columns that actually exist
features = [
    column
    for column in features
    if column in df.columns
]

X = df[features].copy()
y = df[target].copy()

print("\n" + "=" * 70)
print("MODEL VARIABLES")
print("=" * 70)

print("\nFeatures used:")

for feature in features:
    print(f"- {feature}")

print(f"\nTarget: {target}")

# ------------------------------------------------------------
# 6. Correctly identify numerical and categorical columns
# ------------------------------------------------------------
# IMPORTANT:
# We use is_numeric_dtype() instead of checking
# dtype == "object".
#
# This works correctly with modern pandas string dtypes.

categorical_features = [
    column
    for column in features
    if not is_numeric_dtype(X[column])
]

numerical_features = [
    column
    for column in features
    if is_numeric_dtype(X[column])
]

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)

print("\nCategorical features:")

for feature in categorical_features:
    print(f"- {feature}")

print("\nNumerical features:")

for feature in numerical_features:
    print(f"- {feature}")

# ------------------------------------------------------------
# 7. Numerical preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

# ------------------------------------------------------------
# 8. Categorical preprocessing
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# ------------------------------------------------------------
# 9. Combine preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# ------------------------------------------------------------
# 10. Create regression model
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

# ------------------------------------------------------------
# 11. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Training records: {len(X_train):,}"
)

print(
    f"Testing records : {len(X_test):,}"
)

# ------------------------------------------------------------
# 12. Train regression model
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING REGRESSION MODEL")
print("=" * 70)

try:

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed successfully.")

except Exception as e:

    print("\nERROR DURING MODEL TRAINING:")
    print(e)
    raise

# ------------------------------------------------------------
# 13. Generate predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)

print("Predictions generated successfully.")

# ------------------------------------------------------------
# 14. Evaluate model
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

# ------------------------------------------------------------
# 15. Display model performance
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REGRESSION MODEL PERFORMANCE")
print("=" * 70)

print(
    f"MAE  : {mae:.4f} days"
)

print(
    f"RMSE : {rmse:.4f} days"
)

print(
    f"R²   : {r2:.4f}"
)

# ------------------------------------------------------------
# 16. Interpretation guide
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("METRIC INTERPRETATION")
print("=" * 70)

print(
    "\nMAE = average absolute prediction error in days."
)

print(
    "RMSE = prediction error measure that gives more weight "
    "to larger errors."
)

print(
    "R² = proportion of variation in the target explained "
    "by the model."
)

# ------------------------------------------------------------
# 17. Save model metrics
# ------------------------------------------------------------

metrics = pd.DataFrame({
    "Metric": [
        "Mean Absolute Error",
        "Root Mean Squared Error",
        "R-squared"
    ],

    "Value": [
        mae,
        rmse,
        r2
    ]
})

metrics_file = (
    CHARTS_DIR /
    "regression_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)

# ------------------------------------------------------------
# 18. Save prediction sample
# ------------------------------------------------------------

prediction_results = X_test.copy()

prediction_results[
    "Actual Shipping Days"
] = y_test.values

prediction_results[
    "Predicted Shipping Days"
] = y_pred

prediction_file = (
    CHARTS_DIR /
    "regression_predictions.csv"
)

prediction_results.to_csv(
    prediction_file,
    index=False
)

# ------------------------------------------------------------
# 19. Completion message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REGRESSION ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput files:")

print(
    f"- {metrics_file}"
)

print(
    f"- {prediction_file}"
)

print("\nEverything was saved inside:")
print(CHARTS_DIR)