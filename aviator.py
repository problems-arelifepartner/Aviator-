import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Simulate or Load Historical Data
def generate_synthetic_data(num_entries=1000):
    np.random.seed(42)
    data = {
        "round": np.arange(1, num_entries + 1),
        "multiplier": np.round(np.random.exponential(scale=2.0, size=num_entries), 2)
    }
    return pd.DataFrame(data)

# Load historical data
data = generate_synthetic_data()
print("Sample Data:\n", data.head())

# Step 2: Feature Engineering
def create_features(df):
    df["prev_multiplier"] = df["multiplier"].shift(1)
    df["prev_2_multiplier"] = df["multiplier"].shift(2)
    df["prev_3_multiplier"] = df["multiplier"].shift(3)
    df.dropna(inplace=True)
    return df

data = create_features(data)
print("Data with Features:\n", data.head())

# Step 3: Split Data into Training and Testing Sets
X = data[["prev_multiplier", "prev_2_multiplier", "prev_3_multiplier"]]
y = data["multiplier"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Machine Learning Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Multipliers")
plt.ylabel("Predicted Multipliers")
plt.title("Actual vs Predicted Multipliers")
plt.show()

# Step 6: Predict the Next Multiplier
def predict_next_multiplier(model, recent_multipliers):
    recent_multipliers = np.array(recent_multipliers).reshape(1, -1)
    prediction = model.predict(recent_multipliers)
    return prediction[0]

# Example: Predict the next multiplier
recent_multipliers = [1.8, 2.5, 3.1]  # Replace with real recent multipliers
next_multiplier = predict_next_multiplier(model, recent_multipliers)
print(f"Predicted Next Multiplier: {next_multiplier:.2f}")
