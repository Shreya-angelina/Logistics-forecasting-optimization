import numpy as np
import pandas as pd

np.random.seed(42)

N = 800
dates = pd.date_range("2025-01-01", "2025-12-31", periods=N)

regions = np.random.choice(["North", "South", "East", "West"], N)
carriers = np.random.choice(["Carrier A", "Carrier B", "Carrier C", "Carrier D"], N)
modes = np.random.choice(["Road", "Rail", "Air", "Sea"], N,
                         p=[0.45, 0.25, 0.10, 0.20])

distance = np.random.uniform(50, 2500, N)
volume = np.random.randint(1, 101, N)
warehouse = np.random.uniform(1, 10, N)
fuel = np.random.uniform(2, 18, N)

speed = {"Road": 55, "Rail": 70, "Air": 650, "Sea": 35}
mode_speed = np.array([speed[m] for m in modes])

delivery_time = (
    distance / mode_speed
    + warehouse
    + np.random.normal(0, 2.2, N)
)
delivery_time = np.maximum(delivery_time, 1)

df = pd.DataFrame({
    "OrderDate": dates,
    "Region": regions,
    "Carrier": carriers,
    "TransportMode": modes,
    "DistanceKM": distance.round(2),
    "ShipmentVolumeUnits": volume,
    "WarehouseHandlingHrs": warehouse.round(2),
    "FuelSurchargePct": fuel.round(2),
    "DeliveryTimeHrs": delivery_time.round(2)
})

df.to_csv("logistics_data.csv", index=False)
print("logistics_data.csv created successfully.")
