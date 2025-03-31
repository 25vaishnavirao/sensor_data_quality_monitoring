import numpy as np
import pandas as pd

# Generate synthetic sensor data with noise and missing values
np.random.seed(42)
num_samples = 1000

data = {
    "temperature": np.random.normal(loc=25, scale=5, size=num_samples),
    "humidity": np.random.normal(loc=50, scale=10, size=num_samples),
    "pressure": np.random.normal(loc=1013, scale=15, size=num_samples),
}

df = pd.DataFrame(data)

# Introduce missing values
df.iloc[::20, 0] = np.nan  # Missing values in temperature
df.iloc[::25, 1] = np.nan  # Missing values in humidity

# Save the data
df.to_csv("sensor_data.csv", index=False)
print("Synthetic sensor data generated and saved as 'sensor_data.csv'.")
