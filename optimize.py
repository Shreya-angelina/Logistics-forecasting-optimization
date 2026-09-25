import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    ))
])

model.fit(X, y)

sample = df.sample(200, random_state=42).reset_index(drop=True)

modes = ["Road", "Rail", "Air", "Sea"]

cost_per_km = {
    "Road": 2.2,
    "Rail": 1.5,
    "Air": 5.5,
    "Sea": 0.9
}

capacity = {
    "Road": 120,
    "Rail": 60,
    "Air": 30,
    "Sea": 70
}

SLA_HOURS = 24
n = len(sample)

time_matrix = np.zeros((n, 4))
cost_matrix = np.zeros((n, 4))

for j, mode in enumerate(modes):
    what_if = sample[numeric_features + categorical_features].copy()
    what_if["TransportMode"] = mode

    time_matrix[:, j] = model.predict(what_if)
    cost_matrix[:, j] = (
        sample["DistanceKM"].to_numpy()
        * cost_per_km[mode]
    )

BIG_COST = 1e9
cost_lp = np.where(
    time_matrix <= SLA_HOURS,
    cost_matrix,
    BIG_COST
)

# Each shipment must be assigned exactly one mode.
A_eq = np.zeros((n, n * 4))

for i in range(n):
    A_eq[i, i * 4:(i + 1) * 4] = 1

b_eq = np.ones(n)

# Capacity constraints.
A_ub = np.zeros((4, n * 4))
b_ub = np.array([capacity[m] for m in modes])

for j in range(4):
    A_ub[j, j::4] = 1

result = linprog(
    c=cost_lp.flatten(),
    A_ub=A_ub,
    b_ub=b_ub,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=[(0, 1)] * (n * 4),
    method="highs"
)

if not result.success:
    raise RuntimeError(result.message)

assignment = np.argmax(
    result.x.reshape(n, 4),
    axis=1
)

sample["OptimizedMode"] = [
    modes[i] for i in assignment
]

sample["PredictedDeliveryTimeHrs"] = (
    time_matrix[np.arange(n), assignment]
)

sample["TransportationCost"] = (
    cost_matrix[np.arange(n), assignment]
)

sample.to_csv("optimized_shipments.csv", index=False)

print("Optimization completed successfully.")
print("Total Cost: $", round(sample["TransportationCost"].sum(), 2))
print(
    "Average Delivery Time:",
    round(sample["PredictedDeliveryTimeHrs"].mean(), 2),
    "hours"
)
print(
    "SLA Compliance:",
    round(
        (sample["PredictedDeliveryTimeHrs"] <= SLA_HOURS).mean() * 100,
        2
    ),
    "%"
)

print("\nOptimized Mode Allocation:")
print(sample["OptimizedMode"].value_counts())
