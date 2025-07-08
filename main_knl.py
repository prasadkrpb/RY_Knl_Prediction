
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data():
    df = pd.read_csv("Kurnool.csv")
    drop_cols = [
        'latitude', 'longitude', 'rainfall_seasonal_anomaly', 'pevpr_avg_may_oct',
        'tcdc_avg_may_oct', 'tmax_avg_may_oct', 'dswrf_avg_may_oct', 'soilw_avg_may_oct'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df

def train_and_save_models():
    df = load_data()
    feature_cols = [
        'rhum_avg_may_oct', 'spei_avg_may_oct', 'tmax_max_may_oct',
        'tmin_avg_may_oct', 'tmin_min_may_oct', 'wspd_avg_may_oct',
        'rainfall_sum_may_oct', 'rainfall_iav', 'year'
    ]
    target_col = 'RICE_YIELD_Kgperha'
    X, y = df[feature_cols], df[target_col]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_r2 = cross_val_score(rf_model, X, y, scoring='r2', cv=cv).mean()
    rf_rmse = np.sqrt(-cross_val_score(rf_model, X, y, scoring='neg_mean_squared_error', cv=cv)).mean()
    rf_model.fit(X, y)
    joblib.dump(rf_model, "rf_model.pkl")

    # XGBoost
    xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
    xgb_r2 = cross_val_score(xgb_model, X, y, scoring='r2', cv=cv).mean()
    xgb_rmse = np.sqrt(-cross_val_score(xgb_model, X, y, scoring='neg_mean_squared_error', cv=cv)).mean()
    xgb_model.fit(X, y)
    joblib.dump(xgb_model, "xgb_model.pkl")

    # Save metrics
    metrics = {
        "RandomForest": {"R2": rf_r2, "RMSE": rf_rmse, "feature_importances": rf_model.feature_importances_.tolist()},
        "XGBoost": {"R2": xgb_r2, "RMSE": xgb_rmse, "feature_importances": xgb_model.feature_importances_.tolist()},
        "features": feature_cols
    }
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Correlation Heatmap
    corr_matrix = df[feature_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()

    # Feature Importance Plots
    rf_importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
    rf_importances.plot(kind="barh", color="skyblue", figsize=(8,6))
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png")
    plt.close()

    xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()
    xgb_importances.plot(kind="barh", color="green", figsize=(8,6))
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance_xgb.png")
    plt.close()

    # VIF Table
    X_vif = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).dropna()
    X_vif = X_vif.assign(const=1)
    vif_data = pd.DataFrame({
        "Feature": X_vif.columns[:-1],
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns) - 1)]
    })
    vif_data.to_csv("vif_table.csv", index=False)

    print("✅ Models, metrics, and analysis artifacts saved.")

if __name__ == "__main__":
    train_and_save_models()
