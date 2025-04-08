import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#  Data processing
df = pd.read_csv('tata_steel_data.csv', sep=None, engine='python')
df.columns = ['datetime', 'OpenPrice', 'Highprice', 'lowprice', 'closeprice', 'tradedvalue', 'Numberoftrades', 'TradedQuantity']
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df = df[df['datetime'] >= '2021-01-01']
df = df.sort_values(by='datetime')#DD-MM-YYYY

# Normalize the Open and Close Prices
scaler = MinMaxScaler(feature_range=(0, 1))
df[['OpenPrice_scaled', 'closeprice_scaled']] = scaler.fit_transform(df[['OpenPrice', 'closeprice']])

# Create Sequences for LST
sequence_length = 100 
X, y = [], []

for i in range(sequence_length, len(df) - 1):  # Leave the last row for prediction
    X.append(df[['closeprice_scaled', 'OpenPrice_scaled']].values[i-sequence_length:i])
    y.append(df['OpenPrice_scaled'].values[i + 1])  # Next day's Open price

X, y = np.array(X), np.array(y) 

# Step 4: Build LSTM Model
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 2)),  # More neurons for better learning
    Dropout(0.2),
    LSTM(units=100, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)  # Predict Open Price
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train Model (More epochs for better learning)
model.fit(X, y, epochs=100, batch_size=32)

# Evaluate Model
loss = model.evaluate(X, y)
print(f"Model Loss (Mean Squared Error): {loss:.4f}")



# Step 6: Predict Tomorrow’s Open Price
last_sequence = df[['closeprice_scaled', 'OpenPrice_scaled']].values[-sequence_length:]
last_sequence = np.reshape(last_sequence, (1, sequence_length, 2))

predicted_open_scaled = model.predict(last_sequence)
predicted_open = scaler.inverse_transform([[predicted_open_scaled[0][0], 0]])[0][0]  # Convert back to actual price

# Print Prediction
print(f"📈 Predicted Tata Steel Opening Price for Tomorrow: ₹{predicted_open:.2f}")


