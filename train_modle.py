import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import json

# ----------------------------- #
# Generate synthetic dataset
n_samples = 2000
data = pd.DataFrame({
    "username": ["user" + str(i) for i in range(n_samples)],
    "followers": np.random.randint(0, 5000, n_samples),
    "following": np.random.randint(0, 2000, n_samples),
    "posts": np.random.randint(0, 300, n_samples),
    "bio_length": np.random.randint(0, 200, n_samples),
    "has_profile_pic": np.random.choice([0,1], n_samples, p=[0.1,0.9]),
    "external_url": np.random.choice([0,1], n_samples, p=[0.7,0.3]),
    "account_age_days": np.random.randint(1, 3000, n_samples),
    "activity_score": np.random.randint(0, 101, n_samples)
})

# ----------------------------- #
# Label generation using rules
def label_row(row):
    score = 0
    if row["followers"] < 50: score += 1
    if row["following"] > 500: score += 1
    if row["posts"] < 5: score += 1
    if row["bio_length"] < 10: score += 1
    if row["has_profile_pic"] == 0: score += 1
    if row["external_url"] == 0: score += 1
    if row["account_age_days"] < 60: score += 1
    if row["activity_score"] < 30: score += 1
    return 1 if score >= 3 else 0

data["label"] = data.apply(label_row, axis=1)

# ----------------------------- #
# Balance dataset
df_majority = data[data.label == 0]
df_minority = data[data.label == 1]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# ----------------------------- #
# Username features
def extract_username_features(username):
    username = str(username)
    length = len(username)
    letters = sum(c.isalpha() for c in username)
    digits = sum(c.isdigit() for c in username)
    has_name = 1 if letters > digits else 0
    return length, has_name

df_balanced["username_length"], df_balanced["has_name"] = zip(*df_balanced["username"].apply(extract_username_features))

# ----------------------------- #
# Features and labels
features = [
    "username_length", "has_name", "followers", "following", "posts",
    "bio_length", "account_age_days", "activity_score",
    "has_profile_pic", "external_url"
]

X = df_balanced[features]
y = df_balanced["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# ----------------------------- #
# Train SVM model
from sklearn.svm import SVC

svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train_scaled, y_train)
joblib.dump(svm_model, "svm_model.pkl")

# ----------------------------- #
# Train ANN model
ann_model = Sequential([
    Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_scaled, y_train, epochs=30, verbose=0)
ann_model.save("ann_model.h5")

# ----------------------------- #
# Compute accuracies
accuracies = {}
accuracies["SVM"] = round(accuracy_score(y_test, svm_model.predict(X_test_scaled)) * 100, 2)
accuracies["ANN"] = round(accuracy_score(y_test, (ann_model.predict(X_test_scaled) > 0.5).astype(int)) * 100, 2)

# Save accuracies
with open("model_accuracies.json", "w") as f:
    json.dump(accuracies, f)

print("Training completed successfully!")
print("Model Accuracies:", accuracies)
