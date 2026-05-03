import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Sample data creation
data = {
    'location': ['Downtown', 'Suburb', 'Downtown', 'Suburb', 'Rural'],
    'amenities': ['Wifi,Kitchen', 'Wifi,Parking', 'Kitchen,Parking', 'Wifi,Kitchen,Parking', 'Wifi'],
    'bedrooms': [2, 3, 1, 2, 1],
    'price': [150, 120, 100, 130, 80]
}

df = pd.DataFrame(data)

# Preprocess amenities
df['amenities'] = df['amenities'].apply(lambda x: x.split(','))

# One-hot encode location and amenities
ohe_location = OneHotEncoder(sparse=False)
location_encoded = ohe_location.fit_transform(df[['location']])

ohe_amenities = OneHotEncoder(sparse=False)
amenities_encoded = ohe_amenities.fit_transform(df['amenities'].apply(lambda x: ','.join(x)).to_frame())

import numpy as np
X = np.hstack([location_encoded, amenities_encoded, df[['bedrooms']].values])
y = df['price'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction
example = np.hstack([ohe_location.transform([['Downtown']]), ohe_amenities.transform([['Wifi,Kitchen']]), [[2]]])
predicted_price = model.predict(example)
print(f"Predicted price for example listing: ${predicted_price[0]:.2f}")
