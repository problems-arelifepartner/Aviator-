# Load historical Aviator game data
def load_actual_data(file_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    return data

# Replace the synthetic data with actual data
data = load_actual_data("aviator_data.csv")
print("Sample Data:\n", data.head())
