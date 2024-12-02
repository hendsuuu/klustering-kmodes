from flask import Flask, jsonify, request
import pickle
import pandas as pd
import json
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

latest_data = pd.DataFrame()

try:
    cluster_model = pickle.load(open('cluster_model.pkl', 'rb'))
    rules = pickle.load(open('rules_model.pkl', 'rb'))
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari body request
        data = request.get_json()
        
        print(data)

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Konversi data JSON ke DataFrame
        df = pd.DataFrame([data])
        df = apply_thresholds(df)
        # print(df)
        df = cluster(df, cluster_model)
        df['Binary Pattern'], df['Readable Pattern'] = zip(*df.apply(get_pattern, axis=1))
        
        # print(df)

        # Proses hasil prediksi
        pattern = []
        warning = []

        for i in range(len(df)):
            binary_pattern = df['Binary Pattern'][i]
            if binary_pattern in rules:
                pattern.append(binary_pattern)
                warning.append(1)
            else:
                warning.append(0)
        
        for i in warning:
            if i == 1:
                send_mqtt_message('Kualitas Air Baik')
            else:
                send_mqtt_message('Kualitas Air Buruk')

        # Kirimkan response
        response = {
            "data": df.to_dict(orient="records"),
            "warnings": warning,
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

def apply_thresholds(df):
    column_thresholds = {
        'Suhu': (25.0, 30.0),
        'pH': (6.0, 9.0),
        'Amonia': (0.0, 0.8)
    }

    for column, thresholds in column_thresholds.items():
        lower_bound, upper_bound = thresholds

        # Convert the column to numeric values
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Apply thresholds
        df[f'{column.lower()}_thresholded'] = df[column].apply(lambda val: 1 if (val is not None) and (val > upper_bound or val < lower_bound) else 0)
    return df

def get_pattern(raw_features):
    pattern_binary = ''
    pattern_readable = ''

    for i in range(3):
        feature_name = raw_features.index[i]
        is_failure = raw_features.iloc[i + 3] == 1
        pattern_readable += f'{feature_name}: {"Failure" if is_failure else "Normal"}, '
        pattern_binary += str(int(raw_features.iloc[i + 3]))

    return pattern_binary, pattern_readable

def cluster(df, loaded_model):
    df['Condition'] = loaded_model.predict(df[['suhu_thresholded', 'ph_thresholded', 'amonia_thresholded']])
    return df

def send_mqtt_message(warning):
    print(f'Katup {warning}')
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)