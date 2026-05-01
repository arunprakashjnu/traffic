"""
=============================================================
  AI/ML CSV Analysis Suite
  Covers: Descriptive Stats, Correlation, Regression,
          Classification prep, Clustering, PCA, and more.
=============================================================
Requirements:
    pip install pandas numpy scipy scikit-learn matplotlib seaborn
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    silhouette_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

DIVIDER = "\n" + "═" * 60 + "\n"

def section(title: str):
    print(DIVIDER + f"  {title}" + DIVIDER)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅  Loaded '{path}'  →  {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def categorical_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ─────────────────────────────────────────────
# 1. BASIC OVERVIEW
# ─────────────────────────────────────────────

def basic_overview(df: pd.DataFrame):
    section("1. BASIC OVERVIEW")
    print("Shape:", df.shape)
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())
    print("\nBasic info:")
    df.info()


# ─────────────────────────────────────────────
# 2. MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────

def missing_value_analysis(df: pd.DataFrame):
    section("2. MISSING VALUE ANALYSIS")
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    report = report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    if report.empty:
        print("✅  No missing values found.")
    else:
        print(report)
    return report


# ─────────────────────────────────────────────
# 3. DESCRIPTIVE STATISTICS (Mean, Median, Mode, etc.)
# ─────────────────────────────────────────────

def descriptive_statistics(df: pd.DataFrame):
    section("3. DESCRIPTIVE STATISTICS")
    num = numeric_cols(df)
    if not num:
        print("No numeric columns found.")
        return

    print("─── Standard describe() ───")
    print(df[num].describe().T.round(4))

    print("\n─── Mean ───")
    print(df[num].mean().round(4))

    print("\n─── Median ───")
    print(df[num].median().round(4))

    print("\n─── Mode (first) ───")
    print(df[num].mode().iloc[0].round(4))

    print("\n─── Variance ───")
    print(df[num].var().round(4))

    print("\n─── Standard Deviation ───")
    print(df[num].std().round(4))

    print("\n─── Skewness ───")
    print(df[num].skew().round(4))

    print("\n─── Kurtosis ───")
    print(df[num].kurtosis().round(4))

    print("\n─── Range (max – min) ───")
    print((df[num].max() - df[num].min()).round(4))

    print("\n─── IQR (Q3 – Q1) ───")
    Q1 = df[num].quantile(0.25)
    Q3 = df[num].quantile(0.75)
    print((Q3 - Q1).round(4))


# ─────────────────────────────────────────────
# 4. FREQUENCY DISTRIBUTION (Categorical)
# ─────────────────────────────────────────────

def frequency_distribution(df: pd.DataFrame):
    section("4. FREQUENCY DISTRIBUTION (Categorical)")
    cats = categorical_cols(df)
    if not cats:
        print("No categorical columns found.")
        return
    for col in cats:
        print(f"\n── {col} ──")
        vc = df[col].value_counts()
        pct = df[col].value_counts(normalize=True).round(4) * 100
        print(pd.DataFrame({"Count": vc, "Percent %": pct}))


# ─────────────────────────────────────────────
# 5. OUTLIER DETECTION (IQR + Z-Score)
# ─────────────────────────────────────────────

def outlier_detection(df: pd.DataFrame):
    section("5. OUTLIER DETECTION")
    num = numeric_cols(df)
    if not num:
        print("No numeric columns.")
        return

    print("─── IQR Method ───")
    for col in num:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        print(f"  {col}: {len(outliers)} outliers  (bounds [{lower:.3f}, {upper:.3f}])")

    print("\n─── Z-Score Method (|z| > 3) ───")
    z_scores = np.abs(stats.zscore(df[num].dropna()))
    z_df = pd.DataFrame(z_scores, columns=num)
    for col in num:
        n = (z_df[col] > 3).sum()
        print(f"  {col}: {n} outliers")


# ─────────────────────────────────────────────
# 6. CORRELATION ANALYSIS
# ─────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame):
    section("6. CORRELATION ANALYSIS")
    num = numeric_cols(df)
    if len(num) < 2:
        print("Need ≥ 2 numeric columns.")
        return

    corr = df[num].corr()
    print("Pearson Correlation Matrix:\n", corr.round(3))

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(num)), max(5, len(num) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    print("📊  Saved: correlation_heatmap.png")
    plt.close()

    # Highly correlated pairs
    print("\n─── Highly Correlated Pairs (|r| ≥ 0.75, excluding diagonal) ───")
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.75:
                pairs.append((corr.columns[i], corr.columns[j], round(r, 4)))
    if pairs:
        for a, b, r in pairs:
            print(f"  {a}  ↔  {b}  :  r = {r}")
    else:
        print("  None found.")


# ─────────────────────────────────────────────
# 7. PROBABILITY DISTRIBUTIONS & NORMALITY TESTS
# ─────────────────────────────────────────────

def normality_tests(df: pd.DataFrame):
    section("7. NORMALITY TESTS (Shapiro-Wilk)")
    num = numeric_cols(df)
    for col in num:
        data = df[col].dropna()
        if len(data) < 3:
            continue
        sample = data.sample(min(5000, len(data)), random_state=42)
        stat, p = stats.shapiro(sample)
        result = "✅ Normal" if p > 0.05 else "❌ Not Normal"
        print(f"  {col}: W={stat:.4f}, p={p:.4f}  →  {result}")


# ─────────────────────────────────────────────
# 8. HYPOTHESIS TESTING
# ─────────────────────────────────────────────

def hypothesis_testing(df: pd.DataFrame):
    section("8. HYPOTHESIS TESTING (One-sample t-test vs population mean = 0)")
    num = numeric_cols(df)
    for col in num:
        data = df[col].dropna()
        if len(data) < 2:
            continue
        t_stat, p = stats.ttest_1samp(data, popmean=0)
        print(f"  {col}: t={t_stat:.4f}, p={p:.4f}  →  {'Reject H₀' if p < 0.05 else 'Fail to Reject H₀'}")

    # ANOVA between groups (if any categorical × numeric pair exists)
    cats = categorical_cols(df)
    num = numeric_cols(df)
    if cats and num:
        cat_col = cats[0]
        num_col = num[0]
        groups = [grp[num_col].dropna().values for _, grp in df.groupby(cat_col)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            f_stat, p = stats.f_oneway(*groups)
            print(f"\n─── One-Way ANOVA: {num_col} by {cat_col} ───")
            print(f"  F={f_stat:.4f}, p={p:.4f}  →  {'Significant' if p < 0.05 else 'Not Significant'}")


# ─────────────────────────────────────────────
# 9. DATA PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    section("9. DATA PREPROCESSING")
    df = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    cats = categorical_cols(df)
    for col in cats:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Label-encoded: {col}")

    # Impute missing numeric
    num = numeric_cols(df)
    if df[num].isnull().any().any():
        imp = SimpleImputer(strategy="mean")
        df[num] = imp.fit_transform(df[num])
        print("  Imputed missing numeric values with column mean.")

    print("  ✅ Preprocessing complete.")
    return df


# ─────────────────────────────────────────────
# 10. FEATURE SCALING
# ─────────────────────────────────────────────

def feature_scaling(df: pd.DataFrame):
    section("10. FEATURE SCALING")
    num = numeric_cols(df)
    if not num:
        return None, None

    scaler_std = StandardScaler()
    scaler_mm = MinMaxScaler()

    std_scaled = pd.DataFrame(scaler_std.fit_transform(df[num]), columns=num)
    mm_scaled = pd.DataFrame(scaler_mm.fit_transform(df[num]), columns=num)

    print("Standard Scaled (first 3 rows):\n", std_scaled.head(3).round(4))
    print("\nMin-Max Scaled (first 3 rows):\n", mm_scaled.head(3).round(4))
    return std_scaled, mm_scaled


# ─────────────────────────────────────────────
# 11. LINEAR REGRESSION
# ─────────────────────────────────────────────

def linear_regression(df: pd.DataFrame):
    section("11. LINEAR REGRESSION")
    num = numeric_cols(df)
    if len(num) < 2:
        print("Need ≥ 2 numeric columns.")
        return

    target = num[-1]
    features = num[:-1]
    X = df[features].dropna()
    y = df[target].loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  Target column: '{target}'")
    print(f"  Features: {features}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {np.sqrt(mse):.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Coefficients: {dict(zip(features, model.coef_.round(4)))}")
    print(f"  Intercept:    {model.intercept_:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"  5-Fold CV R²: {cv_scores.round(4)}  →  Mean: {cv_scores.mean():.4f}")


# ─────────────────────────────────────────────
# 12. CLASSIFICATION (Random Forest + Logistic Regression)
# ─────────────────────────────────────────────

def classification(df: pd.DataFrame):
    section("12. CLASSIFICATION")
    num = numeric_cols(df)
    if len(num) < 2:
        print("Need ≥ 2 numeric columns.")
        return

    target = num[-1]
    features = num[:-1]
    X = df[features].dropna()
    y_raw = df[target].loc[X.index]

    # Convert to binary class by median split
    median_val = y_raw.median()
    y = (y_raw >= median_val).astype(int)
    print(f"  Target: '{target}' binarized at median ({median_val:.4f})")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print("\n─── Logistic Regression ───")
    print(classification_report(y_test, lr_pred))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("─── Random Forest ───")
    print(classification_report(y_test, rf_pred))

    # Feature importance
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("Feature Importances:\n", imp.round(4).to_string())

    # Confusion Matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, pred, name in zip(axes, [lr_pred, rf_pred], ["Logistic Regression", "Random Forest"]):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix – {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    print("📊  Saved: confusion_matrices.png")
    plt.close()


# ─────────────────────────────────────────────
# 13. CLUSTERING (K-Means)
# ─────────────────────────────────────────────

def clustering(df: pd.DataFrame):
    section("13. CLUSTERING (K-Means)")
    num = numeric_cols(df)
    if len(num) < 2:
        print("Need ≥ 2 numeric columns.")
        return

    X = df[num].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    k_range = range(2, min(11, len(X)))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig("kmeans_elbow.png", dpi=150)
    print("📊  Saved: kmeans_elbow.png")
    plt.close()

    # Fit with k=3
    best_k = 3 if len(X) >= 3 else 2
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  K={best_k} → Silhouette Score: {sil:.4f}")
    print(f"  Cluster sizes: {pd.Series(labels).value_counts().to_dict()}")


# ─────────────────────────────────────────────
# 14. PCA (Dimensionality Reduction)
# ─────────────────────────────────────────────

def pca_analysis(df: pd.DataFrame):
    section("14. PCA – DIMENSIONALITY REDUCTION")
    num = numeric_cols(df)
    if len(num) < 2:
        print("Need ≥ 2 numeric columns.")
        return

    X = df[num].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(len(num), len(X))
    pca = PCA(n_components=n_comp)
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("Explained Variance Ratio per Component:")
    for i, (ev, cum) in enumerate(zip(explained, cumulative), 1):
        print(f"  PC{i}: {ev:.4f}  (Cumulative: {cum:.4f})")

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, n_comp + 1), explained, label="Individual")
    plt.step(range(1, n_comp + 1), cumulative, where="mid", color="red", label="Cumulative")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_scree.png", dpi=150)
    print("📊  Saved: pca_scree.png")
    plt.close()


# ─────────────────────────────────────────────
# 15. DISTRIBUTION PLOTS
# ─────────────────────────────────────────────

def distribution_plots(df: pd.DataFrame):
    section("15. DISTRIBUTION PLOTS")
    num = numeric_cols(df)
    if not num:
        return

    cols = num[:6]  # limit to 6
    n = len(cols)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for ax_row, col in zip(axes, cols):
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax_row[0], color="steelblue")
        ax_row[0].set_title(f"Histogram – {col}")
        sns.boxplot(x=data, ax=ax_row[1], color="coral")
        ax_row[1].set_title(f"Boxplot – {col}")

    plt.tight_layout()
    plt.savefig("distributions.png", dpi=150)
    print("📊  Saved: distributions.png")
    plt.close()


# ─────────────────────────────────────────────
# 16. RANDOM FOREST REGRESSION
# ─────────────────────────────────────────────

def rf_regression(df: pd.DataFrame):
    section("16. RANDOM FOREST REGRESSION")
    num = numeric_cols(df)
    if len(num) < 2:
        print("Need ≥ 2 numeric columns.")
        return

    target = num[-1]
    features = num[:-1]
    X = df[features].dropna()
    y = df[target].loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Target: '{target}'")
    print(f"  RMSE: {np.sqrt(mse):.4f}   R²: {r2:.4f}")

    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("Feature Importances:\n", imp.round(4).to_string())


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_all(csv_path: str):
    df = load_csv(csv_path)

    basic_overview(df)
    missing_value_analysis(df)
    descriptive_statistics(df)
    frequency_distribution(df)
    outlier_detection(df)
    correlation_analysis(df)
    normality_tests(df)
    hypothesis_testing(df)

    df_clean = preprocess(df)
    feature_scaling(df_clean)
    distribution_plots(df_clean)
    linear_regression(df_clean)
    classification(df_clean)
    clustering(df_clean)
    pca_analysis(df_clean)
    rf_regression(df_clean)

    print(DIVIDER + "  ✅  ALL EXERCISES COMPLETE\n" + "═" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python aiml_analysis.py  your_data.csv")
        sys.exit(1)
    run_all(sys.argv[1])
