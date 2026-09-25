import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("logistics_data.csv")
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["OrderMonth"] = df["OrderDate"].dt.month
df["OrderDayOfWeek"] = df["OrderDate"].dt.dayofweek

numeric_features = [
    "DistanceKM",
    "ShipmentVolumeUnits",
    "WarehouseHandlingHrs",
    "FuelSurchargePct",
    "OrderMonth",
    "OrderDayOfWeek"
]

categorical_features = ["Region", "Carrier", "TransportMode"]

X = df[numeric_features + categorical_features]
y = df["DeliveryTimeHrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)

    results.append([
        name,
        mean_squared_error(y_test, prediction) ** 0.5,
        mean_absolute_error(y_test, prediction),
        r2_score(y_test, prediction)
    ])

comparison = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_leaf": [1, 3, 5]
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

prediction = best_rf.predict(X_test)

print("\nBest Parameters:")
print(grid.best_params_)

print("\nTuned Random Forest:")
print("RMSE:", round(mean_squared_error(y_test, prediction) ** 0.5, 3))
print("MAE :", round(mean_absolute_error(y_test, prediction), 3))
print("R2  :", round(r2_score(y_test, prediction), 3))

cv_rmse = -cross_val_score(
    best_rf,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("\n5-Fold CV RMSE:",
      round(cv_rmse.mean(), 3),
      "+/-",
      round(cv_rmse.std(), 3))

# Actual vs predicted
plt.figure(figsize=(8, 7))
plt.scatter(y_test, prediction, alpha=0.65)
lims = [
    min(y_test.min(), prediction.min()),
    max(y_test.max(), prediction.max())
]
plt.plot(lims, lims, "--")
plt.xlabel("Actual Delivery Time (hrs)")
plt.ylabel("Predicted Delivery Time (hrs)")
plt.title("Actual vs Predicted Delivery Time")
plt.tight_layout()
plt.savefig("m1_actual_vs_predicted.png", dpi=180)
plt.show()

# Residuals
residuals = y_test - prediction

plt.figure(figsize=(9, 6))
plt.scatter(prediction, residuals, alpha=0.65)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Delivery Time (hrs)")
plt.ylabel("Residual")
plt.title("Residual Analysis")
plt.tight_layout()
plt.savefig("m2_residuals.png", dpi=180)
plt.show()

# Feature importance
feature_names = best_rf.named_steps["prep"].get_feature_names_out()
importance = best_rf.named_steps["model"].feature_importances_

feature_importance = (
    pd.Series(importance, index=feature_names)
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(9, 6))
feature_importance.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("m3_feature_importance.png", dpi=180)
plt.show()

comparison.to_csv("model_comparison.csv", index=False)
