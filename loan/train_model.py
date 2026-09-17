import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.inspection import permutation_importance

# Load data
df = pd.read_csv("Loan_default.csv")
df = df.drop(columns=["LoanID"])

# Engineered features (small but real signal boost)
df["LoanToIncome"] = df["LoanAmount"] / df["Income"]
df["IncomePerCreditLine"] = df["Income"] / (df["NumCreditLines"] + 1)

# Encode categoricals
cat_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose",
            "HasMortgage", "HasDependents", "HasCoSigner"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Split features/target
X = df.drop(columns=["Default"])
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
num_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
            "LoanToIncome", "IncomePerCreditLine"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# HistGradientBoosting handles this data slightly better than Logistic
# Regression or plain RandomForest (higher ROC-AUC, better precision/recall
# balance once the threshold is tuned)
model = HistGradientBoostingClassifier(
    class_weight="balanced", max_iter=300, random_state=42
)
model.fit(X_train, y_train)

# Find the threshold that maximizes F1 instead of using the default 0.5 cutoff
# (0.5 barely catches any defaults on this imbalanced data)
y_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = f1_scores[:-1].argmax()
best_threshold = float(thresholds[best_idx])

y_pred = (y_proba >= best_threshold).astype(int)
print(f"Best threshold: {best_threshold:.3f}")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# HistGradientBoosting doesn't expose feature_importances_/coef_, so compute
# permutation importance once here and save it for the dashboard to display
sample_idx = X_test.sample(min(5000, len(X_test)), random_state=42).index
perm = permutation_importance(
    model, X_test.loc[sample_idx], y_test.loc[sample_idx],
    n_repeats=3, random_state=42, n_jobs=-1, scoring="roc_auc"
)
feature_importance = pd.Series(perm.importances_mean, index=X.columns) \
    .sort_values(ascending=False)

# Save everything the app needs
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
with open("threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)
with open("feature_importance.pkl", "wb") as f:
    pickle.dump(feature_importance, f)
with open("metrics.pkl", "wb") as f:
    pickle.dump({
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_scores[best_idx],
        "precision": precision[best_idx],
        "recall": recall[best_idx],
        "threshold": best_threshold,
        "n_rows": len(df),
        "default_rate": y.mean(),
    }, f)

print("Saved model.pkl, scaler.pkl, columns.pkl, threshold.pkl, metrics.pkl, feature_importance.pkl")