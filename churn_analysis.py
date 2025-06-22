import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/telco_churn.csv")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categoricals
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feat_importances[:10], y=feat_importances.index[:10])
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")

# 📤 Export predictions for dashboard
X_test['churn_predicted'] = y_pred
X_test.to_csv("data/churn_predictions.csv", index=False)
print("Exported: churn_predictions.csv")
