# water-quality-potability-ml
End-to-end Machine Learning project predicting water potability using Random Forest, XGBoost, KMeans clustering, and SMOTE, Python, Scikit-Learn, SQL

# Water Quality Potability Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=mysql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AA4A44?style=flat&logo=xgboost&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

## Overview

A full end-to-end machine learning project to predict whether water is safe
to drink based on its chemical properties. The project covers SQL-based
exploration, data cleaning, feature engineering, unsupervised clustering,
multi-model classification, and imbalance handling using SMOTE.

---

## Problem Statement

Access to safe drinking water is a critical public health issue. This project
builds a classification model that predicts water potability from measurable
chemical attributes — enabling faster, data-driven water safety assessments
without relying solely on laboratory testing.

**Target variable:** `Potability` — 1 (safe to drink) / 0 (not safe)

---

## Dataset

**Source:** [Kaggle — Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

- 3,276 water samples
- 9 chemical features: pH, Hardness, Solids, Chloramines, Sulfate,
  Conductivity, Organic Carbon, Trihalomethanes, Turbidity
- Class distribution: 61% Not Potable / 39% Potable (imbalanced)

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python (Pandas, NumPy) | Data manipulation and analysis |
| SQLite + SQL | In-notebook data exploration and validation |
| Scikit-Learn | Preprocessing, clustering, classification, evaluation |
| XGBoost | Gradient boosting classification |
| imbalanced-learn (SMOTE) | Synthetic minority oversampling |
| Matplotlib / Seaborn | Visualisation |
| Google Colab | Development environment |

---

## Project Workflow

### Part 1 — Data Foundation
- Loaded dataset into a **SQLite database** and ran SQL queries to
  validate class distribution, detect missing values, and compare average
  chemical levels between potable and non-potable samples
- Confirmed missing values: pH (~15%), Sulfate (~24%), Trihalomethanes (~5%)
- Filled missing values using **grouped median by Potability class** —
  preserves class-specific chemical signal rather than blurring it with
  a global statistic

### Part 2 — Feature Engineering & Encoding
- **pH Category** — binned pH into WHO-based health categories
  (Dangerously Acidic → Safe → Dangerously Alkaline)
- **Hardness-to-Chloramines Ratio** — captures the balance between
  natural mineral content and chemical treatment load
- **Contamination Score** — composite feature counting simultaneous
  threshold violations across multiple chemical readings
- Applied **StandardScaler** — more robust to outliers than MinMaxScaler
  for water chemical measurements

### Part 3 — Clustering Analysis (Unsupervised)
- Tested KMeans for K=2 to K=10 using Elbow curve and Silhouette score
- Selected **K=3** — best balance between cluster quality and interpretability
- Named clusters based on chemical profiles:
  - **Cluster 0:** Balanced Water Group — potability rate 36.6%
  - **Cluster 1:** Alkaline Water Group — highest potability rate 41.8%
  - **Cluster 2:** High Mineral Water Group — lowest potability rate 33.5%
- Visualised clusters using **PCA projection** (2D)
- Key insight: one extreme composition cluster had **0% potability** —
  every sample was non-potable, a pattern invisible to classifiers alone

### Part 4 — Classification (Before Imbalance Handling)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 60.98% | 0.00 | 0.00 | 0.00 | 0.500 |
| Random Forest | 66.46% | 0.645 | 0.313 | 0.421 | 0.601 |
| XGBoost | 60.98% | 0.500 | 0.371 | 0.426 | 0.567 |

> **Key insight:** Logistic Regression completely failed to identify potable
> water samples (Recall = 0, ROC-AUC = 0.5 = random guessing). This confirmed
> the decision boundary is non-linear. Ensemble models significantly
> outperformed the baseline.

### Part 5 — Imbalance Handling

Two strategies were applied to address the 61/39 class imbalance:

**Strategy 1 — Class Weights (Random Forest)**
Applied class weighting to penalise misclassification of the minority class.
Result: marginal improvement only — class weighting alone was insufficient.

**Strategy 2 — SMOTE (Synthetic Minority Oversampling)**
Generated synthetic potable water samples to balance the training set.
Result: improved recall and F1-score, making the model better at detecting
safe water. Selected as the final recommended approach.

**✅ Recommended Model: Random Forest with SMOTE**
Chosen because it achieved the best balance between recall and F1-score
for the positive class — the most critical metric when the objective is
correctly identifying safe drinking water.

---

## Key Findings

- No single chemical feature cleanly separates potable from non-potable
  water — class averages are surprisingly similar, requiring combined
  feature patterns for reliable classification
- Logistic Regression is entirely unsuitable — linear decision boundaries
  cannot capture the complexity of water chemistry data
- XGBoost achieved the highest recall before imbalance handling;
  Random Forest with SMOTE achieved the best post-imbalance performance
- Clustering revealed a 0% potability cluster invisible to classifiers —
  demonstrating the value of combining unsupervised and supervised learning
- Accuracy is a misleading metric on imbalanced data — **Recall and F1
  are the correct evaluation measures for this problem**

---

## Repository Structure

```
├── Water_Quality_Potability.ipynb     # Full project notebook (Google Colab)
├── water_potability.csv               # Raw dataset (source: Kaggle)
└── README.md                          # Project documentation
```

---

## How to Run

1. Open `Water_Quality_Potability.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload your `kaggle.json` API key when prompted to download the dataset
3. Run all cells top to bottom

---

## Author

**Oluwagbotemi Olanrewaju**
Data Analyst | Data Science in Training
[GitHub Profile](https://github.com/YOUR-USERNAME)

> *This project was completed as part of a team capstone assignment.
> All analysis, code, and documentation were produced independently.*
