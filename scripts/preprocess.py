import pandas as pd

def preprocess_data(df):
    """
    Preprocess sensor data:
    - Converts 'timestamp' column to UNIX seconds if it exists.
    - Drops non-numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing sensor data.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with numeric values.
    """

    # Ensure 'timestamp' column exists before conversion
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("int64") // 10**9

    # Drop non-numeric columns (if any)
    df = df.select_dtypes(include=["number"])

    # Drop rows with missing values after conversion
    df = df.dropna()

    return df
