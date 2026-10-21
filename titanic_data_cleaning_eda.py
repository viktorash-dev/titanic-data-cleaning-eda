
"""
PROJECT: Data Cleaning, Preprocessing and EDA
DATASET: Titanic Passenger Dataset
TOOLS: Python, Pandas, NumPy, Matplotlib, Seaborn

Outputs are saved in the titanic_project_outputs folder.
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PROJECT SETUP AND DATASET LOADING
# ============================================================

OUTPUT_DIR = Path("titanic_project_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_URL = "https://hbiostat.org/data/repo/titanic3.csv"
DATA_FILE = OUTPUT_DIR / "titanic3.csv"


def load_dataset():
    """Download the public dataset if necessary and load it."""
    if not DATA_FILE.exists():
        print("Downloading Titanic dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    data = pd.read_csv(
        DATA_FILE,
        na_values=["", "NA", "N/A", "?"]
    )

    # Standardize column names
    data.columns = (
        data.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Convert blank strings to missing values
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip()
        data[column] = data[column].replace("", np.nan)

    return data


df = load_dataset()

print("\n" + "=" * 60)
print("1. INITIAL DATASET INSPECTION")
print("=" * 60)

print("Dataset dimensions:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nDataset structure:")
df.info()

print("\nSummary statistics:")
print(df.describe(include="all").T)


# ============================================================
# 2. DATA QUALITY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("2. DATA QUALITY CHECKS")
print("=" * 60)

missing_before = df.isna().sum().sort_values(ascending=False)

print("\nMissing values by column:")
print(missing_before[missing_before > 0])

print("\nNumber of duplicate rows:", df.duplicated().sum())

missing_before.rename("missing_count").to_csv(
    OUTPUT_DIR / "missing_values_before.csv"
)

# Preserve raw data for comparisons
clean = df.copy()


# ============================================================
# 3. DATA TYPE CONVERSION
# ============================================================

print("\n" + "=" * 60)
print("3. DATA TYPE CONVERSION")
print("=" * 60)

clean["survived"] = clean["survived"].map(
    {0: "No", 1: "Yes"}
).astype("category")

clean["pclass"] = clean["pclass"].map(
    {1: "1st", 2: "2nd", 3: "3rd"}
).astype("category")

for column in ["sex", "embarked"]:
    clean[column] = clean[column].astype("category")

print(clean.dtypes)


# ============================================================
# 4. MISSING VALUE TREATMENT
# ============================================================

print("\n" + "=" * 60)
print("4. MISSING VALUE TREATMENT")
print("=" * 60)

# Median imputation for numeric variables
# Add a flag so the original missingness is not lost.

for column in ["age", "fare"]:
    clean[f"{column}_missing_flag"] = (
        clean[column].isna().astype(int)
    )

    median_value = clean[column].median()

    clean[column] = clean[column].fillna(median_value)

    print(f"{column}: missing values filled with median {median_value:.3f}")

# Explicitly label missing categorical values as Unknown.
for column in ["sex", "pclass", "embarked"]:
    if "Unknown" not in clean[column].cat.categories:
        clean[column] = clean[column].cat.add_categories(["Unknown"])

    clean[column] = clean[column].fillna("Unknown")

print("\nRemaining missing values in key analysis columns:")
print(clean[["age", "fare", "sex", "pclass", "embarked"]].isna().sum())


# ============================================================
# 5. OUTLIER DETECTION USING THE IQR METHOD
# ============================================================

print("\n" + "=" * 60)
print("5. OUTLIER DETECTION")
print("=" * 60)


def detect_outliers(series):
    """Return IQR fences and the number of potential outliers."""
    values = series.dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    count = ((values < lower_fence) | (values > upper_fence)).sum()

    return {
        "variable": series.name,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outlier_count": int(count)
    }


outlier_results = []

for column in ["age", "fare", "sibsp", "parch"]:
    outlier_results.append(detect_outliers(df[column]))

outlier_df = pd.DataFrame(outlier_results)

print(outlier_df.to_string(index=False))

outlier_df.to_csv(
    OUTPUT_DIR / "outlier_summary.csv",
    index=False
)

# Outliers are flagged, not automatically deleted.
# Extreme values may represent genuine passengers or ticket fares.


# ============================================================
# 6. MIN-MAX NORMALIZATION
# ============================================================

print("\n" + "=" * 60)
print("6. NUMERIC NORMALIZATION")
print("=" * 60)

numeric_columns = ["age", "fare", "sibsp", "parch"]

for column in numeric_columns:
    minimum = clean[column].min()
    maximum = clean[column].max()

    if maximum == minimum:
        clean[f"{column}_scaled"] = 0.0
    else:
        clean[f"{column}_scaled"] = (
            (clean[column] - minimum) / (maximum - minimum)
        )

print("Created scaled columns:")
print([f"{column}_scaled" for column in numeric_columns])


# ============================================================
# 7. CATEGORICAL ENCODING
# ============================================================

print("\n" + "=" * 60)
print("7. CATEGORICAL ENCODING")
print("=" * 60)

categorical_columns = ["sex", "pclass", "embarked"]

encoded_df = pd.get_dummies(
    clean[categorical_columns],
    dtype=int
)

print("\nOne-hot encoded data:")
print(encoded_df.head())

encoded_df.to_csv(
    OUTPUT_DIR / "categorical_encoded.csv",
    index=False
)


# ============================================================
# 8. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("8. DESCRIPTIVE STATISTICS")
print("=" * 60)

numeric_summary = clean[numeric_columns].agg(
    ["mean", "median", "std", "min", "max"]
).T

print(numeric_summary)

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_summary.csv"
)

print("\nCategorical summaries:")
print(clean[["survived", "sex", "pclass", "embarked"]].describe().T)


# ============================================================
# 9. SURVIVAL RATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("9. SURVIVAL RATE ANALYSIS")
print("=" * 60)

survival_by_sex = pd.crosstab(
    clean["sex"],
    clean["survived"],
    normalize="index"
)

survival_by_class = pd.crosstab(
    clean["pclass"],
    clean["survived"],
    normalize="index"
)

print("\nSurvival proportion by sex:")
print(survival_by_sex.round(3))

print("\nSurvival proportion by passenger class:")
print(survival_by_class.round(3))

survival_by_sex.to_csv(
    OUTPUT_DIR / "survival_rate_by_sex.csv"
)

survival_by_class.to_csv(
    OUTPUT_DIR / "survival_rate_by_class.csv"
)


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("10. CORRELATION ANALYSIS")
print("=" * 60)

correlation = df[numeric_columns].corr()

print(correlation.round(3))

correlation.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Matrix of Numeric Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()


# ============================================================
# 11. VISUALIZATIONS
# ============================================================

print("\n" + "=" * 60)
print("11. GENERATING VISUALIZATIONS")
print("=" * 60)

sns.set_theme(style="whitegrid")

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"].dropna(), bins=25, kde=True)
plt.title("Passenger Age Distribution")
plt.xlabel("Age in years")
plt.ylabel("Number of passengers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=150)
plt.close()

# Fare outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"])
plt.title("Fare Distribution and Potential Outliers")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fare_boxplot.png", dpi=150)
plt.close()

# Survival by sex
if "Yes" in survival_by_sex.columns:
    survival_by_sex["Yes"].plot(
        kind="bar",
        figsize=(7, 5),
        ylim=(0, 1),
        title="Survival Rate by Sex",
        ylabel="Survival proportion"
    )
    plt.xlabel("Sex")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_by_sex.png", dpi=150)
    plt.close()

# Survival by passenger class
if "Yes" in survival_by_class.columns:
    survival_by_class["Yes"].plot(
        kind="bar",
        figsize=(7, 5),
        ylim=(0, 1),
        title="Survival Rate by Passenger Class",
        ylabel="Survival proportion"
    )
    plt.xlabel("Passenger class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "survival_by_class.png", dpi=150)
    plt.close()


# ============================================================
# 12. EXPORT CLEANED DATA AND FINAL LOG
# ============================================================

print("\n" + "=" * 60)
print("12. EXPORTING RESULTS")
print("=" * 60)

clean.to_csv(
    OUTPUT_DIR / "titanic_cleaned.csv",
    index=False
)

with open(
    OUTPUT_DIR / "analysis_log.txt",
    "w",
    encoding="utf-8"
) as log:
    log.write(f"Dataset dimensions: {df.shape}\n")
    log.write(f"Duplicate rows: {df.duplicated().sum()}\n")

    log.write("\nMissing values before cleaning:\n")
    log.write(missing_before.to_string())

    log.write("\n\nOutlier summary:\n")
    log.write(outlier_df.to_string(index=False))

    log.write("\n\nNumeric summary:\n")
    log.write(numeric_summary.to_string())

    log.write("\n\nCorrelation matrix:\n")
    log.write(correlation.to_string())

print("Analysis completed successfully.")
print("All outputs are saved in:", OUTPUT_DIR.resolve())

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)
