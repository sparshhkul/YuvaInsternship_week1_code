import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# ============================================================
# YUVAINTERN WEEK 1 - FAST CLUSTERING ANALYSIS
# ============================================================

print("=" * 70)
print("YUVAINTERN WEEK 1 - FAST CLUSTERING ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "DataCo_Logistics_Cleaned.csv"
)

CHARTS_DIR = os.path.join(BASE_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

# Remove cancelled records because they do not represent
# completed delivery performance.
df = df[df["Delivery Status"] != "Shipping canceled"].copy()

print(f"Records used: {len(df):,}")


# ------------------------------------------------------------
# 3. SELECT CLUSTERING FEATURES
# ------------------------------------------------------------

features = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Delay",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount",
    "Order Item Product Price"
]

# Check that all required columns exist
missing_columns = [col for col in features if col not in df.columns]

if missing_columns:
    print("\nERROR: The following columns are missing:")
    for col in missing_columns:
        print("-", col)
    raise SystemExit


cluster_data = df[features].copy()


# ------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ------------------------------------------------------------

cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan)

cluster_data = cluster_data.fillna(
    cluster_data.median(numeric_only=True)
)


# ------------------------------------------------------------
# 5. STANDARDIZE DATA
# ------------------------------------------------------------

print("\nStandardizing data...")

scaler = StandardScaler()

X = scaler.fit_transform(cluster_data)

print("Standardization completed.")


# ------------------------------------------------------------
# 6. PREPARE SAMPLE FOR CLUSTER EVALUATION
# ------------------------------------------------------------

print("\nPreparing sample for cluster evaluation...")

SAMPLE_SIZE = min(10000, len(X))

np.random.seed(42)

sample_indices = np.random.choice(
    len(X),
    size=SAMPLE_SIZE,
    replace=False
)

X_sample = X[sample_indices]

print(f"Sample used for evaluation: {SAMPLE_SIZE:,} records")


# ------------------------------------------------------------
# 7. TEST DIFFERENT NUMBERS OF CLUSTERS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TESTING CLUSTER COUNTS")
print("=" * 70)

cluster_results = []

best_k = None
best_score = -1

for k in range(2, 6):

    print(f"\nTesting K = {k}...")

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=2048,
        n_init=3
    )

    sample_labels = model.fit_predict(X_sample)

    score = silhouette_score(
        X_sample,
        sample_labels
    )

    cluster_results.append({
        "Number of Clusters": k,
        "Silhouette Score": round(score, 4)
    })

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )

    if score > best_score:
        best_score = score
        best_k = k


# ------------------------------------------------------------
# 8. SAVE CLUSTER EVALUATION
# ------------------------------------------------------------

evaluation_df = pd.DataFrame(cluster_results)

evaluation_path = os.path.join(
    CHARTS_DIR,
    "clustering_evaluation.csv"
)

evaluation_df.to_csv(
    evaluation_path,
    index=False
)


# ------------------------------------------------------------
# 9. BEST CLUSTER CONFIGURATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST CLUSTER CONFIGURATION")
print("=" * 70)

print(f"Best number of clusters: {best_k}")
print(f"Best silhouette score: {best_score:.4f}")


# ------------------------------------------------------------
# 10. TRAIN FINAL CLUSTERING MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING FINAL CLUSTERING MODEL")
print("=" * 70)

print("Training on complete dataset...")

final_model = MiniBatchKMeans(
    n_clusters=best_k,
    random_state=42,
    batch_size=2048,
    n_init=3
)

cluster_labels = final_model.fit_predict(X)

print("Final clustering completed.")


# ------------------------------------------------------------
# 11. ADD CLUSTER LABELS
# ------------------------------------------------------------

cluster_data_with_labels = cluster_data.copy()

cluster_data_with_labels["Cluster"] = cluster_labels


# ------------------------------------------------------------
# 12. CLUSTER SUMMARY
# ------------------------------------------------------------

cluster_summary = (
    cluster_data_with_labels
    .groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\n" + "=" * 70)
print("COMPLETE CLUSTER SUMMARY")
print("=" * 70)

# This prints ALL columns without pandas truncating them.
print(cluster_summary.to_string())


# ------------------------------------------------------------
# 13. SAVE CLUSTER SUMMARY
# ------------------------------------------------------------

cluster_summary_path = os.path.join(
    CHARTS_DIR,
    "cluster_summary.csv"
)

cluster_summary.to_csv(
    cluster_summary_path
)


# ------------------------------------------------------------
# 14. CLUSTER SIZES
# ------------------------------------------------------------

cluster_sizes = (
    cluster_data_with_labels["Cluster"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("CLUSTER SIZES")
print("=" * 70)

print(cluster_sizes.to_string())


# ------------------------------------------------------------
# 15. SAVE CLUSTER SIZES
# ------------------------------------------------------------

cluster_sizes_df = cluster_sizes.reset_index()

cluster_sizes_df.columns = [
    "Cluster",
    "Number of Records"
]

cluster_sizes_path = os.path.join(
    CHARTS_DIR,
    "cluster_sizes.csv"
)

cluster_sizes_df.to_csv(
    cluster_sizes_path,
    index=False
)


# ------------------------------------------------------------
# 16. CLUSTER PERCENTAGES
# ------------------------------------------------------------

cluster_percentages = (
    cluster_sizes / len(cluster_data_with_labels) * 100
).round(2)

print("\n" + "=" * 70)
print("CLUSTER PERCENTAGES")
print("=" * 70)

for cluster, percentage in cluster_percentages.items():
    print(
        f"Cluster {cluster}: "
        f"{percentage:.2f}%"
    )


# ------------------------------------------------------------
# 17. SAVE CLUSTERED SAMPLE
# ------------------------------------------------------------

# Save a manageable sample instead of the complete
# 172k-row dataset.

output_sample_size = min(
    10000,
    len(cluster_data_with_labels)
)

clustered_sample = (
    cluster_data_with_labels
    .sample(
        n=output_sample_size,
        random_state=42
    )
)

clustered_sample_path = os.path.join(
    CHARTS_DIR,
    "clustered_logistics_data_sample.csv"
)

clustered_sample.to_csv(
    clustered_sample_path,
    index=False
)


# ------------------------------------------------------------
# 18. CREATE CLUSTER VISUALIZATION
# ------------------------------------------------------------

print("\nCreating cluster visualization...")

import matplotlib.pyplot as plt

plot_size = min(
    15000,
    len(cluster_data_with_labels)
)

plot_data = (
    cluster_data_with_labels
    .sample(
        n=plot_size,
        random_state=42
    )
)

plt.figure(figsize=(10, 6))

for cluster in sorted(plot_data["Cluster"].unique()):

    cluster_points = plot_data[
        plot_data["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_points["Days for shipping (real)"],
        cluster_points["Sales"],
        label=f"Cluster {cluster}",
        alpha=0.5,
        s=15
    )

plt.xlabel("Actual Shipping Days")

plt.ylabel("Sales")

plt.title(
    "Logistics Customer/Order-Item Clusters"
)

plt.legend()

plt.tight_layout()

chart_path = os.path.join(
    CHARTS_DIR,
    "09_logistics_clusters.png"
)

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Cluster chart saved to:\n{chart_path}"
)


# ------------------------------------------------------------
# 19. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print("- clustering_evaluation.csv")
print("- cluster_summary.csv")
print("- cluster_sizes.csv")
print("- clustered_logistics_data_sample.csv")
print("- 09_logistics_clusters.png")

print("\nAll files are saved inside:")
print(CHARTS_DIR)

print("\n" + "=" * 70)
print("IMPORTANT: COMPLETE CLUSTER SUMMARY ABOVE")
print("=" * 70)