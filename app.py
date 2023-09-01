


import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import json

# Load your trained model
with open('pickles/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_multiple.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded Excel file
    excel_file = request.files['file']
    if not excel_file:
        return "No file uploaded", 400

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(excel_file)

    # Store the IDs and drop the ID column
    ids = df['ID']
    feature_columns = [
        'original_firstorder_90Percentile', 'original_firstorder_Entropy', 'original_firstorder_Maximum',
        'original_glcm_Correlation', 'original_glcm_Imc1', 'original_glszm_ZoneEntropy',
        'log-sigma-1-0-mm-3D_firstorder_Mean', 'log-sigma-1-0-mm-3D_glrlm_RunEntropy',
        'log-sigma-1-0-mm-3D_glrlm_RunLengthNonUniformityNormalized', 'log-sigma-1-0-mm-3D_glszm_ZoneEntropy',
        'wavelet-LLH_firstorder_Mean'
    ]
    feature_df = df[feature_columns]

    # Make a prediction using the loaded model
    predictions = model.predict(feature_df.values)

    # Map the prediction results to labels
    label_mapping = {0: 'Benign', 1: 'Malignant'}
    prediction_labels = [label_mapping[pred] for pred in predictions]

    # Combine the IDs and the prediction labels into a dictionary
    results = {id_: label for id_, label in zip(ids, prediction_labels)}

    # Save the prediction results to a JSON file with each result in a new line
    with open('predictions.json', 'w') as json_file:
        json.dump(results, json_file, separators=(',', ':'), indent=4)

    # Return the prediction results
    return render_template('predict.html', results=json.dumps(results, separators=(',', ':'), indent=4))





if __name__ == "__main__":
    # Use the port specified in the WEBSITE_PORT environment variable or fallback to 8000
    port = int(os.environ.get('WEBSITE_PORT', 8050))
    app.run(port=port, debug=True)













