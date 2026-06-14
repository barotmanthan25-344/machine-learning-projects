import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ==================================
# LOAD DATASET
# ==================================
df = pd.read_csv(r"C:\Users\manthan barot\OneDrive\Desktop\WELTECH\MACHINE LEARNING\ML TEST - 13.6.26\ml_datasets\dbscan_dataset.csv")

# ==================================
# SELECT FEATURES
# ==================================
X = df[["annual_spend_inr",
        "txn_frequency_per_year"]]

# ==================================
# FEATURE SCALING
# ==================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================
# DBSCAN WITH EPS = 0.3
# ==================================
dbscan1 = DBSCAN(
    eps=0.3,
    min_samples=5
)

clusters1 = dbscan1.fit_predict(X_scaled)

# ==================================
# VISUALIZATION
# ==================================
plt.figure(figsize=(6,5))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters1
)

plt.title("DBSCAN (eps = 0.3)")
plt.xlabel("Annual Spend")
plt.ylabel("Transaction Frequency")

plt.show()

# ==================================
# DBSCAN WITH EPS = 0.5
# ==================================
dbscan2 = DBSCAN(
    eps=0.5,
    min_samples=5
)

clusters2 = dbscan2.fit_predict(X_scaled)

plt.figure(figsize=(6,5))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters2
)

plt.title("DBSCAN (eps = 0.5)")
plt.xlabel("Annual Spend")
plt.ylabel("Transaction Frequency")

plt.show()

# ==================================
# DBSCAN WITH EPS = 0.8
# ==================================
dbscan3 = DBSCAN(
    eps=0.8,
    min_samples=5
)

clusters3 = dbscan3.fit_predict(X_scaled)

plt.figure(figsize=(6,5))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters3
)

plt.title("DBSCAN (eps = 0.8)")
plt.xlabel("Annual Spend")
plt.ylabel("Transaction Frequency")

plt.show()

# ==================================
# COMPARISON
# ==================================
print("eps = 0.3")
print("Clusters Found:",
      len(set(clusters1)) -
      (1 if -1 in clusters1 else 0))

print("\neps = 0.5")
print("Clusters Found:",
      len(set(clusters2)) -
      (1 if -1 in clusters2 else 0))

print("\neps = 0.8")
print("Clusters Found:",
      len(set(clusters3)) -
      (1 if -1 in clusters3 else 0))