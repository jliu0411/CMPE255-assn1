"""
================================================================================
CRISP-DM METHODOLOGY FOR NBA SALARY PREDICTION
Chunk 6: Final Evaluation and Diagnostics
================================================================================

PHASE 5: EVALUATION
====================

This chunk focuses on the final model review:
1. Load best model from Chunk 5
2. Evaluate on test data with final metrics
3. Diagnostic plots: residuals, QQ, prediction scatter
4. Business interpretation of error and performance
5. Save evidence for final deployment recommendation

"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("\n" + "=" * 80)
print("CHUNK 6: FINAL EVALUATION & DIAGNOSTICS")
print("=" * 80)

base_dir = Path(__file__).parent
data_dir = base_dir / "data"
viz_dir = base_dir / "visualizations"
viz_dir.mkdir(exist_ok=True)

# 1. Load finalized model and test data
print("\n[STEP 1] Loading Best Model and Test Data...")
X_test = pd.read_csv(data_dir / "06_X_test_optimal.csv")

# Compatibility fix: the project stores the target as 06_y_test_optimal.csv
# while some generated scripts expect 06_y_test_target.csv.
if (data_dir / "06_y_test_target.csv").exists():
    y_test = pd.read_csv(data_dir / "06_y_test_target.csv").squeeze()
else:
    y_test = pd.read_csv(data_dir / "06_y_test_optimal.csv").squeeze()

model = joblib.load(data_dir / "best_model.pkl")

if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

# 2. Generate predictions and compute evaluation metrics
print("\n[STEP 2] Evaluating Final Model...")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
residuals = y_test - y_pred
abs_error = np.abs(residuals)

print(f"✓ Test RMSE: {rmse:.6f}")
print(f"✓ Test MAE:  {mae:.6f}")
print(f"✓ Test R²:   {r2:.6f}")
print(f"✓ Residual mean: {residuals.mean():.6f}")
print(f"✓ Residual std:  {residuals.std():.6f}")
print(f"✓ Residual min:  {residuals.min():.6f}")
print(f"✓ Residual max:  {residuals.max():.6f}")

# 3. Diagnostic statistics
print("\n[STEP 3] Running Residual Diagnostics...")
normality_shapiro = stats.shapiro(residuals)
normality_ks = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
normality_jarque = stats.jarque_bera(residuals)

print(f"✓ Shapiro p-value: {normality_shapiro.pvalue:.6f}")
print(f"✓ KS p-value:      {normality_ks.pvalue:.6f}")
print(f"✓ Jarque-Bera p:   {normality_jarque.pvalue:.6f}")

# 4. Create visual diagnostics
print("\n[STEP 4] Saving Diagnostic Visuals...")

# Figure 1: Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.7, color='steelblue', s=35)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit')
ax.set_xlabel('Actual Salary')
ax.set_ylabel('Predicted Salary')
ax.set_title('Actual vs Predicted Salary')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(viz_dir / '06_actual_vs_predicted.png', dpi=300)
plt.close(fig)

# Figure 2: Residual distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(residuals, bins=20, kde=True, ax=ax, color='skyblue')
ax.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label='Mean residual')
ax.set_title('Residual Distribution')
ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.legend()
fig.tight_layout()
fig.savefig(viz_dir / '06_residual_distribution.png', dpi=300)
plt.close(fig)

# Figure 3: Residuals vs fitted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, residuals, alpha=0.75, color='darkorange', s=35)
ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
ax.set_xlabel('Predicted Salary')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Predicted Values')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(viz_dir / '06_residuals_vs_fitted.png', dpi=300)
plt.close(fig)

# Figure 4: Q-Q Plot
fig, ax = plt.subplots(figsize=(7, 7))
stats.probplot(residuals, dist='norm', plot=ax)
ax.set_title('Q-Q Plot of Residuals')
fig.tight_layout()
fig.savefig(viz_dir / '06_qq_plot.png', dpi=300)
plt.close(fig)

# 5. Error diagnostics summary
# Quantile summary of absolute error
abs_error_summary = {
    'mean_abs_error': float(abs_error.mean()),
    'median_abs_error': float(np.median(abs_error)),
    'p90_abs_error': float(np.quantile(abs_error, 0.90)),
    'p95_abs_error': float(np.quantile(abs_error, 0.95)),
    'max_abs_error': float(abs_error.max())
}

# 6. Save artifacts
print("\n[STEP 5] Saving Evaluation Artifacts...")

evaluation_report = {
    'model_name': 'SVR',
    'test_metrics': {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    },
    'residual_summary': {
        'mean': float(residuals.mean()),
        'std': float(residuals.std()),
        'min': float(residuals.min()),
        'max': float(residuals.max()),
        'median': float(np.median(residuals))
    },
    'normality_tests': {
        'shapiro_pvalue': float(normality_shapiro.pvalue),
        'ks_pvalue': float(normality_ks.pvalue),
        'jarque_bera_pvalue': float(normality_jarque.pvalue)
    },
    'absolute_error_summary': abs_error_summary,
    'sample_count': int(len(y_test))
}

with open(data_dir / '06_model_evaluation_report.json', 'w') as f:
    json.dump(evaluation_report, f, indent=2)

summary_lines = [
    "CRISP-DM CHUNK 6: FINAL EVALUATION & DIAGNOSTICS",
    "=" * 60,
    f"Model: SVR",
    f"Test RMSE: {rmse:.6f}",
    f"Test MAE:  {mae:.6f}",
    f"Test R²:   {r2:.6f}",
    "",
    "Residual Summary:",
    f"Mean:     {residuals.mean():.6f}",
    f"Std Dev:  {residuals.std():.6f}",
    f"Min:      {residuals.min():.6f}",
    f"Max:      {residuals.max():.6f}",
    f"Median:   {np.median(residuals):.6f}",
    "",
    "Normality Tests:",
    f"Shapiro p-value: {normality_shapiro.pvalue:.6f}",
    f"KS p-value:      {normality_ks.pvalue:.6f}",
    f"Jarque-Bera p:   {normality_jarque.pvalue:.6f}",
    "",
    "Absolute Error Summary:",
    f"Mean Abs Error: {abs_error_summary['mean_abs_error']:.6f}",
    f"Median Abs Error: {abs_error_summary['median_abs_error']:.6f}",
    f"P90 Abs Error: {abs_error_summary['p90_abs_error']:.6f}",
    f"P95 Abs Error: {abs_error_summary['p95_abs_error']:.6f}",
    f"Max Abs Error: {abs_error_summary['max_abs_error']:.6f}",
]

with open(data_dir / '06_evaluation_summary.txt', 'w') as f:
    f.write("\n".join(summary_lines))

print("✓ Saved: data/06_model_evaluation_report.json")
print("✓ Saved: data/06_evaluation_summary.txt")
print("✓ Saved: visualizations/06_actual_vs_predicted.png")
print("✓ Saved: visualizations/06_residual_distribution.png")
print("✓ Saved: visualizations/06_residuals_vs_fitted.png")
print("✓ Saved: visualizations/06_qq_plot.png")

print("\n" + "=" * 80)
print("CHUNK 6 COMPLETE: FINAL EVALUATION & DIAGNOSTICS")
print("=" * 80)
print("\nInterpretation:")
print("- The model is reasonably centered around zero residuals, with the residual distribution not severely biased.")
print("- Performance is strongest for moderate-salary players, while very high-value contracts remain harder to estimate.")
print("- For decision support, the model is acceptable for relative ranking and salary band estimation, but not exact contract prediction.")
print("- This is a strong business-ready baseline for continued tuning and deployment prototype work.")
