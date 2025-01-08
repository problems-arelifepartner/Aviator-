import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load historical Aviator game data
def load_actual_data(file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    return data

# Feature Engineering
def create_features(df):
    df["prev_multiplier"] = df["multiplier"].shift(1)
    df["prev_2_multiplier"] = df["multiplier"].shift(2)
    df["prev_3_multiplier"] = df["multiplier"].shift(3)
    df.dropna(inplace=True)
    return df

# Main Workflow
def main():
    # Step 1: Load Actual Data
    data = load_actual_data("aviator_data.csv")
    print("Loaded Data:\n", data.head())

    # Step 2: Create Features
    data = create_features(data)
    print("Data with Features:\n", data.head())

    # Step 3: Split Data
    X = data[["prev_multiplier", "prev_2_multiplier", "prev_3_multiplier"]]
    y = data["multiplier"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Step 6: Predict Next Multiplier
    recent_multipliers = [1.8, 2.5, 3.1]  # Replace with real recent multipliers
    next_multiplier = model.predict([recent_multipliers])
    print(f"Predicted Next Multiplier: {next_multiplier[0]:.2f}")

if __name__ == "__main__":
    main()
    
