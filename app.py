from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import tensorflow as tf
from scripts.preprocess import preprocess_data

app = Flask(__name__)

# Load trained model
model_path = "E:\\sensor_data_quality_monitoring\\autoencoder_model.keras"
model = tf.keras.models.load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    df_test = pd.read_csv(file)
    df_test = preprocess_data(df_test)

    # Model prediction
    reconstructed = model.predict(df_test)
    errors = ((df_test - reconstructed) ** 2).mean(axis=1)
    df_test["reconstruction_error"] = errors

    threshold = df_test["reconstruction_error"].quantile(0.95)
    df_test["is_anomaly"] = df_test["reconstruction_error"] > threshold

    results = df_test[["reconstruction_error", "is_anomaly"]].to_html()

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
