"""
Part 8: Model Evaluation Metrics — Complete Examples
Run: python part-08/metrics_examples.py
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, log_loss, brier_score_loss, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, ndcg_score)

print("=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)

y_true  = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
y_pred  = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
y_score = np.array([0.9, 0.8, 0.4, 0.3, 0.1, 0.2, 0.15, 0.05, 0.7, 0.6])

print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision:         {precision_score(y_true, y_pred):.4f}")
print(f"Recall:            {recall_score(y_true, y_pred):.4f}")
print(f"F1:                {f1_score(y_true, y_pred):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
print(f"MCC:               {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"Cohen Kappa:       {cohen_kappa_score(y_true, y_pred):.4f}")
print(f"ROC-AUC:           {roc_auc_score(y_true, y_score):.4f}")
print(f"PR-AUC:            {average_precision_score(y_true, y_score):.4f}")
print(f"Log Loss:          {log_loss(y_true, y_score):.4f}")
print(f"Brier Score:       {brier_score_loss(y_true, y_score):.4f}")

print("\n" + "=" * 60)
print("REGRESSION METRICS")
print("=" * 60)

y_true_r = np.array([100., 200., 300., 400., 500.])
y_pred_r = np.array([110., 195., 285., 420., 490.])

mae  = mean_absolute_error(y_true_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_true_r, y_pred_r))
r2   = r2_score(y_true_r, y_pred_r)
rmsle = np.sqrt(np.mean((np.log1p(y_pred_r) - np.log1p(y_true_r))**2))
mape = np.mean(np.abs((y_true_r - y_pred_r) / y_true_r)) * 100
smape = np.mean(np.abs(y_true_r - y_pred_r) /
                ((np.abs(y_true_r) + np.abs(y_pred_r)) / 2)) * 100

def huber_loss(y, y_hat, delta=1.0):
    r = np.abs(y - y_hat)
    return np.where(r <= delta, 0.5*r**2, delta*(r - 0.5*delta)).mean()

print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")
print(f"RMSLE: {rmsle:.4f}")
print(f"MAPE:  {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
print(f"R²:    {r2:.4f}")
print(f"Huber (delta=15): {huber_loss(y_true_r, y_pred_r, delta=15):.2f}")

print("\n" + "=" * 60)
print("RANKING METRICS")
print("=" * 60)

# NDCG
y_rel  = np.array([[3, 2, 3, 0, 1, 2]])
y_scr  = np.array([[2, 3, 1, 0, 2, 3]])
print(f"NDCG@6: {ndcg_score(y_rel, y_scr, k=6):.4f}")

def precision_at_k(y_true, y_pred, k):
    top_k = np.argsort(-y_pred)[:k]
    return y_true[top_k].sum() / k

def recall_at_k(y_true, y_pred, k):
    top_k = np.argsort(-y_pred)[:k]
    return y_true[top_k].sum() / y_true.sum()

def hit_rate_at_k(y_true, y_pred, k):
    top_k = np.argsort(-y_pred)[:k]
    return float(y_true[top_k].sum() > 0)

def mrr(y_true, y_pred):
    ranked = np.argsort(-y_pred)
    for i, idx in enumerate(ranked):
        if y_true[idx] == 1:
            return 1.0 / (i + 1)
    return 0.0

y_true_bin = np.array([1, 0, 1, 0, 1, 0, 0, 1])
y_scores   = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

print(f"P@3:  {precision_at_k(y_true_bin, y_scores, 3):.4f}")
print(f"R@3:  {recall_at_k(y_true_bin, y_scores, 3):.4f}")
print(f"HR@3: {hit_rate_at_k(y_true_bin, y_scores, 3):.4f}")
print(f"MRR:  {mrr(y_true_bin, y_scores):.4f}")

print("\n" + "=" * 60)
print("THOMPSON SAMPLING (from Part 5)")
print("=" * 60)

np.random.seed(42)
true_rates  = [0.10, 0.12, 0.09]

alpha = np.ones(3)
beta  = np.ones(3)
arm_counts = np.zeros(3)

for _ in range(10_000):
    samples = np.random.beta(alpha, beta)
    arm = np.argmax(samples)
    reward = np.random.binomial(1, true_rates[arm])
    alpha[arm] += reward
    beta[arm]  += 1 - reward
    arm_counts[arm] += 1

print(f"Traffic: {arm_counts / arm_counts.sum()}")
# Best arm (12%) should receive ~80%+ of traffic
assert arm_counts[1] / arm_counts.sum() > 0.6, "Thompson sampling not finding best arm"
print("Thompson Sampling correctly identifies the best arm.")

print("\nAll Part 8 examples ran successfully!")
