import pandas as pd
import numpy as np

# Number of samples
n_samples = 2000

# Generate synthetic features
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

# Define fake/real label based on simple rules
def label_row(row):
    score = 0
    if row["followers"] < 50: score += 1
    if row["following"] > 500: score += 1
    if row["posts"] < 5: score += 1
    if row["bio_length"] < 10: score += 1
    if row["has_profile_pic"] == 0: score += 1
    if row["external_url"] == 0: score += 1
    if row["account_age_days"] < 30: score += 1
    if row["activity_score"] < 20: score += 1
    return 1 if score >= 4 else 0

data["label"] = data.apply(label_row, axis=1)

# Shuffle and split
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Save CSVs
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("Train and test CSV files generated successfully!")
