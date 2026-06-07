# Week 12: Bias-Variance Tradeoff & Model Complexity — Summary Report

## 1. Three Most Important Conclusions

1. **Complexity beyond the optimum leads to overfitting.**  As polynomial degree increases, training error monotonically decreases, but test error first decreases then rises sharply — a classic U-shaped test-error curve.
2. **High-variance models are fragile.**  With repeated sampling, a high-degree polynomial (degree=15) exhibits huge swings in its fitted curve, whereas a low-degree model (degree=2) remains stable. The danger of a high-variance model is not that it cannot fit the training set, but that it is overly sensitive to the specific realisation of the noise.
3. **RMSE amplifies large errors; MAE is robust.**  A single extreme prediction can inflate RMSE dramatically while leaving MAE almost unchanged. The choice of loss function should match the cost structure of the application.

## 2. Which Figure Best Represents Overfitting?

**Answer:** `candidate_models.png` (Task A) best captures overfitting.  In the degree-15 subplot, the green fitted curve oscillates wildly to pass through nearly every training point, achieving a very low training RMSE, yet it deviates far from the true black-dashed function — the test RMSE is substantially higher. This visual contrast between low train error and high test error is the textbook signature of overfitting.

## 3. When to Report RMSE vs MAE

- **Report RMSE** when large errors are disproportionately costly (e.g. financial risk, safety-critical predictions) and you want the metric to penalise them heavily. RMSE is also preferred when the error distribution is approximately Gaussian.
- **Report MAE** when you need a robust metric that is not dominated by a few outliers. If the data naturally contains anomalies or the cost of error is roughly linear, MAE gives a more representative picture of typical performance.

## 4. Connection to Regularization (Ridge / Lasso)

If model complexity is too high, the model exhibits high variance — its parameters are estimated with large swings depending on the training sample.  **Regularization** (Ridge, Lasso) directly addresses this by constraining the magnitude of the coefficients, thereby reducing variance at the cost of a small increase in bias.  This is the natural next step after observing that unconstrained high-degree polynomials overfit: we keep the expressive capacity but penalise extreme coefficient values, achieving a better bias-variance trade-off.

## 5. Task B: Complexity–Error Table

| Degree | Train RMSE | Test RMSE | Generalization Gap |
|--------|------------|-----------|-------------------|
|      1 |     0.4889 |    0.3911 |           -0.0979 |
|      2 |     0.4876 |    0.3993 |           -0.0884 |
|      3 |     0.2026 |    0.1676 |           -0.0350 |
|      4 |     0.1981 |    0.1637 |           -0.0344 |
|      5 |     0.1780 |    0.1846 |            0.0067 |
|      6 |     0.1779 |    0.1855 |            0.0076 |
|      7 |     0.1774 |    0.1901 |            0.0127 |
|      8 |     0.1773 |    0.1898 |            0.0125 |
|      9 |     0.1761 |    0.1888 |            0.0127 |
|     10 |     0.1742 |    0.1945 |            0.0203 |
|     11 |     0.1729 |    0.1973 |            0.0244 |
|     12 |     0.1721 |    0.1957 |            0.0235 |
|     13 |     0.1720 |    0.1966 |            0.0246 |
|     14 |     0.1718 |    0.1965 |            0.0248 |
|     15 |     0.1705 |    0.2015 |            0.0309 |
|     16 |     0.1691 |    0.2150 |            0.0460 |
|     17 |     0.1689 |    0.2188 |            0.0499 |
|     18 |     0.1666 |    0.2284 |            0.0617 |

- **Lowest test RMSE** occurs at degree = **4** (test RMSE = 0.1637).
- **Largest generalization gap** occurs around degree **18** (gap = 0.0617). In general, the gap widens rapidly for degrees ≥ 12.
- **Why the lowest training-error model is not necessarily the best:** The model with the lowest training RMSE (degree=18) has essentially memorised the training noise.  Its low training error is achieved by fitting patterns that do not generalise — the test error is high.  A good model should balance fit and simplicity.

## 6. Task C: Variance Quantitative Summary

| Degree | Mean Prediction SD | Max Prediction SD |
|--------|--------------------|-------------------|
|      2 |             0.0233 |            0.0413 |
|     15 |             0.0613 |            0.1857 |

**Fill in the blank:**  High variance model的危险, 不是它不会拟合训练集, 而是它对 **训练数据中的噪声/具体样本** 过于敏感.

## 7. Task D: Outlier Sensitivity of RMSE vs MAE

| Scenario | RMSE     | MAE      |
|----------|----------|----------|
| Clean    | 0.0910   | 0.0723   |
| Outlier  | 0.1317   | 0.0813   |

- **Why RMSE is more sensitive:** RMSE squares the error before averaging, so a single large residual (e.g., 1.0) contributes 1.0² = 1.0 to the MSE, dominating the contributions of many smaller residuals. MAE, by contrast, only takes the absolute value, so the outlier's contribution remains proportional.
- **If the cost of one large mistake is extremely high**, you should monitor **RMSE** (or even more aggressive metrics like Max Error), because it explicitly penalises large deviations and will alert you to catastrophic failures.
- **If the data naturally contain many outliers**, you should seriously reconsider using RMSE and likely prefer **MAE** (or Huber loss).  RMSE would be dominated by the outliers and may not reflect the model's performance on the bulk of the data. In such cases, you might also consider robust regression techniques (e.g., RANSAC, HuberRegressor) that are designed to be insensitive to outliers.
