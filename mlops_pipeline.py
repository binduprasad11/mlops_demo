import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data.csv")
X = df[['weight', 'smooth']]
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")

# Save model
with open("fruit_model.pkl", "wb") as f:
    pickle.dump(model, f)
