# House Prices: Categorical Encoding Techniques for Regression

This repository contains an end-to-end Machine Learning and advanced Feature Engineering notebook that evaluates **six distinct categorical encoding strategies** using the classic Ames Housing dataset. 

Machine learning estimators operate purely on numerical matrices. The choice of how text properties (e.g., neighborhood names, quality rankings, or roof materials) are transformed into numbers shapes the feature space, affects structural dimensions, and determines whether a model is vulnerable to overfitting and data leakage. This project systematically evaluates these adjustments under identical validation constraints.

---

## Core Engineering Pipeline Steps

The project steps in the notebook are executed as follows:

1. **Environment Setup & Libraries:** Initializing dependencies including `pandas`, `numpy`, `matplotlib`, and specific `scikit-learn` diagnostic modules.
2. **Data Ingestion:** Reading raw training variables from disk.
3. **Data Profiling & Diagnostics:** Analyzing shapes, distributions, data types, and identifying structural data gaps (e.g., missing footprints across 19 separate columns).
4. **Target Isolation & Log-Transformation:** Isolating `SalePrice` and applying a `log1p` distribution normalization transformation to optimize regression residuals.
5. **Class/Feature Matrix Split:** Separating predictive variables from target variables early to guarantee clean, leak-free preprocessing.
6. **Unified Missing Value Imputation:** Upfront feature engineering filling text columns with a structural `"Missing"` category and numeric columns with robust medians.
7. **The Modeling Evaluation Harness:** Building a reusable `regress()` utility using an 80/20 `train_test_split` and a deterministic L2 penalized `Ridge` regression model to manage collinearity flags cleanly.
8. **Sequential Method Execution:** Systematically testing out the encoding techniques below, recording out-of-sample Root Mean Squared Error (RMSE), $R^2$, and wall-clock execution timings.

---

## Evaluated Encoding Strategies

| # | Technique | Core Engineering Mechanism | Best Suited For |
|---|-----------|----------------------------|-----------------|
| 1 | **Label Encoding** | Maps each distinct category sequence to an arbitrary unique integer. | Low-level baseline or strictly ordinal features. |
| 2 | **One-Hot Encoding** | Expands text options into independent, sparse binary tracking vectors. | Low-to-medium cardinality nominal groups. |
| 3 | **Feature Hashing** | Uses a deterministic hashing function to map features to a fixed lower-dimensional boundary. | High-cardinality fields or streaming production contexts. |
| 4 | **Count + Ordinal Mappings** | Substitutes category strings with dataset occurrence metrics paired with rank structures. | Compact tabular data layouts. |
| 5 | **Cyclic Encoding** | Transforms periodic parameters (e.g., month sold) into complementary sine/cosine coordinates. | Temporal features with recurring edges. |
| 6 | **Target Encoding (Naive & K-Fold)** | Replaces categories with the target variable mean calculated out-of-fold using multi-split cross-validation. | High-cardinality nominal groups. |

---

## Summary of Experimental Performance Results

The encoding pipelines are compiled and ranked below by their performance on the out-of-sample log-price validation split:

| Strategy Rank | Strategy | RMSE (Log Price) ⬇️ | $R^2$ Score ⬆️ | Encode Time (s) | Fit Time (s) |
|---|---|---|---|---|---|
| 1 | **Target Encoding (Naive)** \* | 0.132792 | 0.905506 | 0.030983 | 0.003527 |
| 2 | **K-Fold Target Encoding** | 0.137000 | 0.899422 | 0.134786 | 0.003404 |
| 3 | **Cyclic (MoSold) + One-Hot** | 0.145258 | 0.886932 | 0.038702 | 0.008962 |
| 4 | **One-Hot Encoding** | 0.148040 | 0.882558 | 0.022245 | 0.009295 |
| 5 | **Dataset-statistics + Ordinal** | 0.150028 | 0.879383 | 0.095709 | 0.003599 |
| 6 | **Label Encoding** | 0.154383 | 0.872279 | 0.023742 | 0.003488 |
| 7 | **Feature Hashing** | 0.187015 | 0.812581 | 0.035240 | 0.007353 |

*\*Note: The naive target encoder exhibits look-ahead data leakage because target records are visible within individual feature transformations, producing artificially optimistic scores. The **K-Fold Target Encoder** acts as the true, robust baseline for practical deployment.*

---

## Key Production Architecture Engineering Takeaways

* **No Universal Encoder Rule:** Selection depends heavily on column cardinality, inherent ordinal traits, matrix volumes, and down-stream learning behaviors. Linear frameworks (like Ridge) remain highly sensitive to encoding representations.
* **Discipline Around Target Leakage:** Naive target values contaminate features with their own historical solutions. In production settings, out-of-fold K-Fold processing is mandatory to secure stable out-of-sample scores.
* **Real-World Missing Data Realities:** Practical data spaces are rarely clean. Meaningful null records (e.g., empty cabin or pool tags indicating their factual absence) must be intentionally preserved as structural flags rather than dropped blindly.

---

## Contributing
Contributions, issue logs, and optimization suggestions are welcome! Feel free to open a pull request or start an issue trail.
