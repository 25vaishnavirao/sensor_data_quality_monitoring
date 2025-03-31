import pandas as pd
import tensorflow as tf
from scripts.preprocess import preprocess_data  

def predict_anomalies(test_csv_path, model_path):
    """
    Loads the trained model and performs anomaly detection on test data.

    Args:
        test_csv_path (str): Path to the test sensor data CSV file.
        model_path (str): Path to the trained model file.

    Returns:
        pd.DataFrame: DataFrame containing anomaly detection results.
    """

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess test data
    df_test = pd.read_csv(test_csv_path)
    df_test = preprocess_data(df_test)  # Converts timestamp & drops non-numeric columns

    # Get predictions
    reconstructed = model.predict(df_test)

    # Compute reconstruction error (anomaly detection)
    errors = ((df_test - reconstructed) ** 2).mean(axis=1)
    df_test["reconstruction_error"] = errors

    # Set an anomaly threshold (adjust as needed)
    threshold = df_test["reconstruction_error"].quantile(0.95)
    df_test["is_anomaly"] = df_test["reconstruction_error"] > threshold

    # Save results
    result_path = "E:\\sensor_data_quality_monitoring\\anomaly_results.csv"
    df_test.to_csv(result_path, index=False)

    print(f"✅ Anomaly detection completed. Results saved in {result_path}")

    return df_test
