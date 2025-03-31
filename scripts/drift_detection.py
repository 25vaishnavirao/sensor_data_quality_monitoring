import pandas as pd

def detect_drift(filepath):
    df = pd.read_csv(filepath)
    drift_threshold = 0.2
    if df.mean().std() > drift_threshold:
        return "🚨 Data drift detected!"
    return "✅ No significant data drift."
