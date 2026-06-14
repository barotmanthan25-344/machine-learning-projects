import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================
# LOAD DATASET
# ==================================
df = pd.read_csv(r"C:\Users\manthan barot\OneDrive\Desktop\WELTECH\MACHINE LEARNING\ML TEST - 13.6.26\ml_datasets\08_svm_diabetes.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# ==================================
# TARGET COLUMN
# ==================================
target_column = df.columns[-1]

X = df.drop(target_column, axis=1)
y = df[target_column]

# ==================================
# HANDLE MISSING VALUES
# ==================================
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# ==================================
# TRAIN TEST SPLIT
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================
# FEATURE SCALING
# ==================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================
# LINEAR KERNEL SVM
# ==================================
linear_model = SVC(kernel='linear')

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_acc = accuracy_score(y_test, linear_pred)

print("\n=================================")
print("LINEAR KERNEL RESULTS")
print("=================================")

print("Accuracy:", round(linear_acc * 100, 2), "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, linear_pred))

print("\nClassification Report")
print(classification_report(y_test, linear_pred))

# ==================================
# RBF KERNEL SVM
# ==================================
rbf_model = SVC(kernel='rbf')

rbf_model.fit(X_train, y_train)

rbf_pred = rbf_model.predict(X_test)

rbf_acc = accuracy_score(y_test, rbf_pred)

print("\n=================================")
print("RBF KERNEL RESULTS")
print("=================================")

print("Accuracy:", round(rbf_acc * 100, 2), "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rbf_pred))

print("\nClassification Report")
print(classification_report(y_test, rbf_pred))

# ==================================
# ACCURACY COMPARISON
# ==================================
print("\n=================================")
print("ACCURACY COMPARISON")
print("=================================")

print("Linear Kernel Accuracy :", round(linear_acc * 100, 2), "%")
print("RBF Kernel Accuracy    :", round(rbf_acc * 100, 2), "%")

if linear_acc > rbf_acc:
    print("\nLinear Kernel Performs Better")
elif rbf_acc > linear_acc:
    print("\nRBF Kernel Performs Better")
else:
    print("\nBoth Kernels Perform Equally")

# ==================================
# ACCURACY GRAPH
# ==================================
kernels = ["Linear", "RBF"]
accuracies = [linear_acc * 100, rbf_acc * 100]

plt.figure(figsize=(6,4))
plt.bar(kernels, accuracies)

plt.title("SVM Kernel Accuracy Comparison")
plt.xlabel("Kernel")
plt.ylabel("Accuracy (%)")

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value:.2f}%")

plt.show()