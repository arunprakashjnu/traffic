import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# CSV Data Source
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'pollution', 'Indicator_3_1_Climate_Indicators_Annual_Mean_Global_Surface_Temperature_6121427861384429071.csv'))

# Load the dataset
# We use pandas to load the dataset. We'll skip invalid rows just in case.
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: Could not find CSV at {CSV_PATH}")
    df = pd.DataFrame()

# Clean and extract countries for dropdown
if not df.empty and 'Country' in df.columns:
    countries_list = df['Country'].dropna().unique().tolist()
    countries_list.sort()
else:
    countries_list = []

# Year arrays
year_cols = [str(y) for y in range(1961, 2025)]
future_years = [str(y) for y in range(2025, 2036)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/countries', methods=['GET'])
def get_countries():
    return jsonify(countries_list)

@app.route('/api/predict', methods=['POST'])
def predict():
    req_data = request.json
    selected_countries = req_data.get('countries', [])
    
    results = {}
    
    for country in selected_countries:
        country_data = df[df['Country'] == country]
        if country_data.empty:
            continue
            
        # Extract temperatures for available years and coerce to numeric
        temps = pd.to_numeric(country_data[year_cols].iloc[0], errors='coerce')
        
        # Drop naive NaN values to create valid X, y pairs for training
        valid_data = temps.dropna()
        if len(valid_data) < 2:
            continue # Not enough data for regression
            
        # Reshape X to 2D array
        X = np.array([int(y) for y in valid_data.index]).reshape(-1, 1)
        y = valid_data.values
        
        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X, y)
        
        # Arrays for plotting
        future_X = np.array([int(y) for y in future_years]).reshape(-1, 1)
        
        # Predictions
        predicted_future = model.predict(future_X)
        
        # Format the data cleanly
        historical_dict = {str(int(year[0])): float(temp) for year, temp in zip(X, y)}
        future_dict = {str(int(year[0])): float(temp) for year, temp in zip(future_X, predicted_future)}
        
        results[country] = {
            'historical': historical_dict,
            'future': future_dict
        }
        
    return jsonify({
        'years': year_cols + future_years,
        'results': results
    })

if __name__ == '__main__':
    app.run(debug=True)
