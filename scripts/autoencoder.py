import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_autoencoder(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Convert 'timestamp' column to numeric format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9  # Convert to seconds

    # Drop non-numeric columns
    df = df.select_dtypes(include=['number'])

    # Normalize data
    df = (df - df.mean()) / df.std()

    # Define Autoencoder Model
    input_dim = df.shape[1]  

    autoencoder = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])

    autoencoder.compile(optimizer="adam", loss="mse")

    # Train the model
    autoencoder.fit(df, df, epochs=50, batch_size=32, validation_split=0.2)

    # Save the trained model
    model_path = "E:\\sensor_data_quality_monitoring\\autoencoder_model.keras"
    autoencoder.save(model_path)  # Use recommended `.keras` format

    return model_path

# Ensure script can run standalone
if __name__ == "__main__":
    csv_path = "E:\\sensor_data_quality_monitoring\\sensor_data.csv"
    train_autoencoder(csv_path)
