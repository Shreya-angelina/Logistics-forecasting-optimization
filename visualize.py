import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("logistics_data.csv")

# Delivery-time distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["DeliveryTimeHrs"], kde=True)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (hours)")
plt.tight_layout()
plt.savefig("delivery_time_distribution.png", dpi=180)
plt.show()

# Mode comparison
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="TransportMode",
    y="DeliveryTimeHrs"
)
plt.title("Delivery Time by Transport Mode")
plt.tight_layout()
plt.savefig("delivery_time_by_mode.png", dpi=180)
plt.show()

# Correlation matrix
numeric = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=180)
plt.show()

# Optimized allocation if output exists
try:
    optimized = pd.read_csv("optimized_shipments.csv")

    allocation = optimized["OptimizedMode"].value_counts()

    plt.figure(figsize=(8, 5))
    allocation.plot(kind="bar")
    plt.xlabel("Transport Mode")
    plt.ylabel("Shipment Count")
    plt.title("LP-Optimized Mode Allocation")
    plt.tight_layout()
    plt.savefig("o2_mode_allocation.png", dpi=180)
    plt.show()

except FileNotFoundError:
    print("Run optimize.py first to create optimized_shipments.csv.")
