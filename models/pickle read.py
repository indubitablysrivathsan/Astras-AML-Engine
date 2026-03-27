import numpy as np
import pandas as pd
import joblib

model = joblib.load("risk_classifier.pkl")
explainer = joblib.load("shap_explainer.pkl")
feature_cols = joblib.load("feature_columns.pkl")

shap_values = explainer.shap_values(X_test)

shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": feature_cols,
    "weight": shap_importance
}).sort_values(by="weight", ascending=False)

print(shap_df.head(20))

shap_values = explainer.shap_values(X_test)

for f, w in zip(feature_cols, shap_importance):
    print(f"{f}: {w:.6f}")