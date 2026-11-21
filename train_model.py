
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("ebp.csv")

# Input features
X = data[
    ["units_consumed", "people", "appliances", "daily_hours"]
]

# Target variable
y = data["bill_amount"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Mean Absolute Error:",
      mean_absolute_error(y_test, predictions))

print("R2 Score:", r2_score(y_test, predictions))

# Save model
with open("electricity_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
