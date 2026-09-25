import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("logistics_data.csv")

df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["OrderMonth"] = df["OrderDate"].dt.month
df["OrderDayOfWeek"] = df["OrderDate"].dt.dayofweek

print("\nDataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nAverage delivery time by transport mode:")
print(df.groupby("TransportMode")["DeliveryTimeHrs"].mean())

plt.figure(figsize=(8, 5))
sns.histplot(df["DeliveryTimeHrs"], kde=True)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (hours)")
plt.tight_layout()
plt.savefig("delivery_time_distribution.png", dpi=180)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="TransportMode", y="DeliveryTimeHrs")
plt.title("Delivery Time by Transport Mode")
plt.tight_layout()
plt.savefig("delivery_time_by_mode.png", dpi=180)
plt.show()

numeric = df.select_dtypes(include="number")
plt.figure(figsize=(10, 7))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=180)
plt.show()
