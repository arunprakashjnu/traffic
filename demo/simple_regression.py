"""
Simple Linear Regression Program
Predicts house prices based on square footage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ===== SAMPLE DATA =====
# House data: Square footage (X) and Price (y)
data = {
    'Square_Feet': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500],
    'Price': [150000, 225000, 300000, 375000, 450000, 525000, 600000, 675000]
}

df = pd.DataFrame(data)

print("=" * 50)
print("SIMPLE LINEAR REGRESSION - HOUSE PRICE PREDICTION")
print("=" * 50)

# ===== PREPARE DATA =====
X = df[['Square_Feet']]  # Features (input)
y = df['Price']           # Target (output)

print("\n📊 Dataset:")
print(df)

# ===== TRAIN MODEL =====
model = LinearRegression()
model.fit(X, y)

# ===== GET RESULTS =====
slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(y, model.predict(X))

print("\n" + "=" * 50)
print("📈 REGRESSION RESULTS:")
print("=" * 50)
print(f"Equation: Price = {slope:.2f} × Square_Feet + {intercept:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Slope: ${slope:.2f} per sq ft")
print(f"Intercept: ${intercept:.2f}")

# ===== PREDICTIONS =====
print("\n" + "=" * 50)
print("🔮 PREDICTIONS:")
print("=" * 50)

test_values = [2200, 3300, 5000]
for sqft in test_values:
    predicted_price = model.predict([[sqft]])[0]
    print(f"  {sqft} sq ft → ${predicted_price:,.2f}")

# ===== VISUALIZATION =====
plt.figure(figsize=(10, 6))

# Plot actual data
plt.scatter(X, y, color='blue', s=100, label='Actual Data', alpha=0.7)

# Plot regression line
X_line = np.array([[X.min()[0]], [X.max()[0]]])
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')

# Plot predictions
for sqft in test_values:
    predicted_price = model.predict([[sqft]])[0]
    plt.scatter([sqft], [predicted_price], color='green', s=150, marker='*', 
                edgecolors='black', linewidth=2, zorder=5)

plt.xlabel('Square Footage', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.title('House Price Prediction - Linear Regression', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simple_regression.png', dpi=150)
print("\n✅ Graph saved as: simple_regression.png")
plt.show()

print("\n" + "=" * 50)
