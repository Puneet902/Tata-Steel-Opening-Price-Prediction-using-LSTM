# 📊 Tata Steel Opening Price Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) deep learning model to predict the **next day's opening price** of Tata Steel stock, based on historical stock data. It uses TensorFlow/Keras and scikit-learn for building and training the model.

---

## 🔧 Features

- Preprocessing and normalization of time-series stock data
- Sequence generation using 100-timestep windows
- LSTM-based neural network model for prediction
- Inverse scaling to get the actual predicted price
- Visualization-ready output
- ✅ Clean and ready-to-use code for retraining or adapting

---

## 🧠 Technologies Used

- Python 🐍
- TensorFlow & Keras
- scikit-learn
- NumPy
- pandas

---

## 📁 Dataset

The dataset should be a CSV file named:

```
tata_steel_data.csv
```

### Format (column order):
| datetime | OpenPrice | Highprice | lowprice | closeprice | tradedvalue | Numberoftrades | TradedQuantity |

- `datetime` format: `DD-MM-YYYY` or compatible with `dayfirst=True`
- Make sure the file is placed in the root directory of the project

---

## 🧪 How It Works

1. **Reads and cleans** historical Tata Steel stock data
2. **Normalizes** the Open and Close prices using MinMaxScaler
3. **Generates sequences** of 100 past days to predict the next day’s Open price
4. **Builds an LSTM model** with two LSTM layers and Dropout regularization
5. **Trains the model** for 100 epochs
6. **Predicts** the next day’s opening price
7. **Denormalizes** the output to get the actual price in rupees

---

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tata-steel-lstm-predictor.git
cd tata-steel-lstm-predictor
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

### `requirements.txt` (example):
```
numpy
pandas
scikit-learn
tensorflow
```

---

## 🚀 Run the Project

Make sure your dataset is in the correct format and placed in the root directory, then run:

```bash
python predict_tata_steel.py
```

---

## 📈 Example Output

```
Model Loss (Mean Squared Error): 0.0007
📊 Predicted Tata Steel Opening Price for Tomorrow: ₹124.52
```

---

## 📝 License

This project is **not open for contribution or redistribution**.

© 2025 Boina Puneet Vaishnav. All rights reserved.

---

## 💡 Future Improvements

- Add support for other features like volume, high/low prices
- Visualization of predicted vs actual prices
- Save and load model for real-time prediction
- Add hyperparameter tuning and model evaluation metrics

---

## 👤 Author

- **Boina Puneet Vaishnav**
- GitHub: [@Puneet902]

---

## 🙌 Contributions

Contributions are **not accepted** for this repository.

