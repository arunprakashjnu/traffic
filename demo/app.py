from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json

app = Flask(__name__)

# ===== TRAIN MODEL ONCE =====
data = {
    'Square_Feet': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
    'Price': [150000, 225000, 300000, 375000, 450000, 525000, 600000, 675000]
}

df = pd.DataFrame(data)
X = df[['Square_Feet']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(y, model.predict(X))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sqft = float(data['sqft'])
    
    predicted_price = model.predict([[sqft]])[0]
    
    return jsonify({
        'sqft': sqft,
        'predicted_price': f"${predicted_price:,.2f}",
        'equation': f"Price = ${slope:.2f} × Square_Feet + ${intercept:.2f}",
        'r2': f"{r2:.4f}"
    })

@app.route('/data')
def get_data():
    return jsonify({
        'sqft': df['Square_Feet'].tolist(),
        'price': df['Price'].tolist(),
        'slope': float(slope),
        'intercept': float(intercept)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
