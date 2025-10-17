# 🏠 House Price Prediction using Linear Regression
# Author: Roshini S T S

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Create a sample dataset
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'price': [2000000, 3000000, 4000000, 4500000, 5000000]
}
df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['area', 'bedrooms']]
y = df['price']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Display results
print("🏡 HOUSE PRICE PREDICTION SYSTEM 🏡\n")
print("Training complete!\n")

# Display predictions
for i in range(len(y_pred)):
    print(f"Predicted: ₹{int(y_pred[i]):,} | Actual: ₹{int(y_test.iloc[i]):,}")

# Step 7: Evaluate the model
print("\nModel Evaluation:")
print("R² Score:", round(r2_score(y_test, y_pred), 3))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))

# Step 8: Custom user input
print("\n--- Try Your Own Prediction ---")
area = float(input("Enter area (in sqft): "))
bedrooms = int(input("Enter number of bedrooms: "))
pred_price = model.predict([[area, bedrooms]])[0]
print(f"Predicted Price: ₹{int(pred_price):,}")
