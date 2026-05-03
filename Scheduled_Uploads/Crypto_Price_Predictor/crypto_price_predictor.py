import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# For demonstration, generate synthetic BTC price data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500)
prices = np.cumsum(np.random.randn(500)) + 10000  # synthetic price data

df = pd.DataFrame({'Date': dates, 'Price': prices})
df.set_index('Date', inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df)

# Prepare data for LSTM
def create_dataset(data, time_step=10):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled_prices, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(X, Y, epochs=10, batch_size=16, verbose=1)

# Predict
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)
original = scaler.inverse_transform(Y.reshape(-1, 1))

# Plot results
plt.plot(original, label='Original Price')
plt.plot(train_predict, label='Predicted Price')
plt.title('BTC Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
