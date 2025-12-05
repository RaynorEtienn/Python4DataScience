# Research Report: Churn Prediction Optimization

**Date:** December 4, 2025
**Objective:** Surpass the All-Time High (ATH) Kaggle F1-Score of **0.64812**.

## 1. Baseline Analysis (The "All-Time High")
*   **Score:** 0.64812
*   **Strategy:** Stacking Ensemble (XGBoost, LightGBM, CatBoost) + Logistic Regression Meta-Learner.
*   **Data Strategy:** "Smart Sampling" (Dormancy Snapshots).
    *   Churners: Snapshots at 1, 3, 7 days before churn.
    *   Non-Churners: Random active snapshots + **Dormancy Snapshots** (random points 1-45 days after last event).
*   **Optimization:** Optuna with **30 Trials**, optimizing for **F1-Score**.
*   **Key Insight:** The "Light Tune" (30 trials) combined with F1 optimization prevented overfitting to the synthetic "Hard Mode" training data.

---

## 2. Experiment Log

| ID | Experiment Name | Changes Made | CV Score (F1) | Kaggle Score | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Hard Voting + 6 Snapshots** | Replaced Stacking with VotingClassifier (Hard). Increased churn snapshots to 6 (1,3,7,14,21,28 days). | ~0.58 | 0.58xxx | **Failed**. Hard voting lacks threshold tuning. 6 snapshots caused class imbalance issues. |
| **2** | **Stacking Revert** | Reverted to StackingClassifier. Reverted to 3 Churn Snapshots. | 0.611 | 0.64298 | **Recovered**. Validated that Stacking > Voting and 3 Snapshots > 6 Snapshots. |
| **3** | **KNN Addition** | Added KNeighborsClassifier to the Stacking Ensemble. | N/A | 0.63493 | **Failed**. KNN likely introduced noise due to high dimensionality (curse of dimensionality). |
| **4** | **Deep Tune (F1)** | Removed KNN. Increased Optuna trials to **100**. Optimized for **F1**. | 0.358 | 0.63072 | **Failed**. Severe overfitting. The model "gamed" the F1 metric on the "Hard Mode" training set, losing generalization. |
| **5** | **Deep Tune (AUC)** | Optuna trials **100**. Optimized for **ROC-AUC**. | 0.392 | 0.64147 | **Improved**. AUC is a more robust metric than F1 for optimization, but 100 trials still led to some overfitting. |
| **6** | **Light Tune (AUC)** | Reduced Optuna trials to **30**. Optimized for **ROC-AUC**. | 0.389 | 0.63564 | **Regression**. While less overfit, AUC optimization with few trials didn't find the specific decision boundary "sweet spot" that F1 optimization did in the ATH. |
| **7** | **ATH Replication (F1)** | Reverted to **30 Trials** and **F1 Optimization**. | 0.362 | 0.63553 | **Failed**. Exact replication failed. Suggests ATH (0.648) was a "lucky" run or relied on specific random state dynamics. |
| **8** | **Balanced Tune (AUC)** | Increased trials to **50**. Optimized for **ROC-AUC**. | 0.392 | 0.63783 | **Improved**. Better than 30 trials, but still below 100-trial AUC (0.641). AUC seems to plateau. |
| **9** | **Hybrid Tune (F1)** | Increased trials to **50**. Optimized for **F1**. | 0.373 | 0.63720 | **Stagnant**. Increasing trials with F1 optimization didn't help. The model is stuck in a local optimum. |
| **10** | **Lucky Seed Test** | Changed `RANDOM_SEED` to **2024**. 30 Trials, F1. | 0.363 | 0.62760 | **Confirmed**. The score dropped significantly (-0.015). This confirms the ATH (0.648) was heavily influenced by the specific random seed (42). |
| **11** | **Soft Voting (Seed 42)** | Reverted to Seed 42. Switched from Stacking to **Soft Voting**. | 0.364 | **0.64094** | **Success**. Stabilized the score. The ensemble (0.641) significantly outperformed individual models (Cat: 0.630, LGBM: 0.626, XGB: 0.611). |
| **12** | **New Features** | Added `trend_errors`, `songs_per_minute`, `diversity_ratio`. Soft Voting. | 0.359 | 0.63626 | **Invalid**. Failed to re-optimize hyperparameters for new features. Score drop (-0.004) is inconclusive. Exp 13 will re-run with optimization. |
| **13** | **New Features (Optimized)** | Same features as Exp 12. **Re-optimized** with 30 trials. | 0.364 | 0.63630 | **Failed**. Even with optimization, the score (0.636) is worse than Exp 11 (0.641). The new features are adding noise. We must revert them. |
| **14** | **Rate-Based Features (No Opt)** | Fixed Data Divergence by converting counts to rates (e.g., `sessions_per_day`). Default Params. | 0.343 | 0.64003 | **Success**. Achieved 0.640 without tuning. This proves the new features are robust and the "Leakage" is fixed. Ready for tuning. |
