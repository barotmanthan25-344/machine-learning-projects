import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# LOAD DATASET

df = pd.read_csv(r"C:\Users\manthan barot\OneDrive\Desktop\WELTECH\MACHINE LEARNING\ML TEST - 13.6.26\ml_datasets\PCA BREST CANCER DATASET.csv")


# DROP USELESS COLUMNS

df.drop("id", axis=1, inplace=True)
df.drop("Unnamed: 32", axis=1, inplace=True)


# ENCODE TARGET COLUMN
# M = 1, B = 0

le = LabelEncoder()

df["diagnosis"] = le.fit_transform(df["diagnosis"])


# FEATURES AND TARGET

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]


# HANDLE MISSING VALUES

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# FEATURE SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# MODEL BEFORE PCA

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model_before = RandomForestClassifier(random_state=42)

model_before.fit(X_train, y_train)

pred_before = model_before.predict(X_test)

acc_before = accuracy_score(y_test, pred_before)


# APPLY PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# PCA VISUALIZATION

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y
)

plt.title("Breast Cancer Dataset PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()


# MODEL AFTER PCA

X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

model_after = RandomForestClassifier(random_state=42)

model_after.fit(X_train_pca, y_train)

pred_after = model_after.predict(X_test_pca)

acc_after = accuracy_score(y_test, pred_after)


# ACCURACY COMPARISON

print("\nAccuracy Before PCA :", round(acc_before*100,2), "%")
print("Accuracy After PCA  :", round(acc_after*100,2), "%")

# COMPARISON GRAPH

models = ["Before PCA", "After PCA"]
accuracies = [acc_before*100, acc_after*100]

plt.figure(figsize=(6,4))

plt.bar(models, accuracies)

plt.title("Model Performance Before and After PCA")
plt.ylabel("Accuracy (%)")

for i, v in enumerate(accuracies):
    plt.text(i, v+1, f"{v:.2f}%")

plt.show()