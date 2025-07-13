# mlops_pipeline.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("data.txt")  # Make sure data.txt is in repo
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save model to file
with open("fruit_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as fruit_model.pkl")

# Predict on a sample input (e.g. 180g, round=1)
sample = [[180, 1]]
prediction = model.predict(sample)[0]
print(f"🍎 Prediction for {sample}: {prediction}")

# Save prediction to a file
with open("prediction.txt", "w") as f:
    f.write(f"Prediction for {sample}: {prediction}")
