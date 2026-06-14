import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# =====================================
# LOAD DATASET
# =====================================
df = pd.read_csv(r"C:\Users\manthan barot\OneDrive\Desktop\WELTECH\MACHINE LEARNING\ML TEST - 13.6.26\ml_datasets\09_kmeans_customer_segmentation.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# =====================================
# HANDLE MISSING VALUES
# =====================================
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# =====================================
# ENCODE CATEGORICAL COLUMNS
# =====================================
le = LabelEncoder()

if "City_Type" in df.columns:
    df["City_Type"] = le.fit_transform(df["City_Type"])

if "Gender" in df.columns:
    df["Gender"] = le.fit_transform(df["Gender"])

# =====================================
# SELECT FEATURES FOR CLUSTERING
# =====================================
X = df[["Purchase_Frequency", "Total_Spend_INR"]]

# =====================================
# FEATURE SCALING
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# ELBOW METHOD
# =====================================
wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

# =====================================
# ELBOW GRAPH
# =====================================
plt.figure(figsize=(7,5))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.show()

# =====================================
# FINAL K-MEANS MODEL
# =====================================
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# =====================================
# CLUSTER VISUALIZATION
# =====================================
plt.figure(figsize=(8,6))

plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=clusters
)

# Cluster Centers
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    marker='X'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spend INR")

plt.show()

# =====================================
# CLUSTER COUNTS
# =====================================
print("\nCluster Distribution")
print(df["Cluster"].value_counts())

# =====================================
# DATA WITH CLUSTER LABELS
# =====================================
print("\nFirst 10 Records")
print(df.head(10))