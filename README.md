# LSTM-Based Temperature Forecasting with STM32 Deployment

*Train a multivariate LSTM to predict temperature using weather data, and deploy it on resource-constrained STM32 microcontrollers.*

> **Project goal** – Use 10 years of hourly weather data to train a deep learning model that can predict temperature without requiring a physical sensor. The final model is optimized and converted to TensorFlow Lite format for embedded deployment.

---

## 🧠 Model Overview

This project uses an LSTM (Long Short-Term Memory) neural network trained on 8 weather features (e.g., humidity, pressure, wind) to predict temperature one hour into the future. Input is based on a one-week rolling window (168 time steps × 8 features).

---

## 📁 Folder structure — top level

```text
.
├── fetch_data.py             ← fetches Meteostat data and saves as CSV
├── temperature_data.csv      ← full 10-year dataset (2015–2025, Hyderabad)
├── process_data.py           ← normalizes and splits data, saves scaler
├── scaler_params.txt         ← saved MinMaxScaler values
├── lstm_model.py             ← builds, trains, and evaluates the LSTM model
├── main.keras                ← saved Keras model after training
├── convert_model.py          ← converts model to TensorFlow Lite format
├── converted_model.tflite    ← lightweight model for deployment
├── README.md                 ← this file
```

---

## 🔧 Installation

```bash
git clone https://github.com/barnav1/PS-I
cd PS-I
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow meteostat
```

*Python 3.8+ recommended.*

---

## 🚀 Running the Pipeline

### Step 1: Fetch Historical Weather Data

```bash
python fetch_data.py
```

Downloads hourly weather data from Meteostat API for Hyderabad (2015–2025).

### Step 2: Preprocess the Data

```bash
python process_data.py
```

- Filters and normalizes the 8 numerical features  
- Saves scaler parameters to `scaler_params.txt`

### Step 3: Train the LSTM Model

```bash
python lstm_model.py
```

- Splits data into train/val/test
- Trains an LSTM model with early stopping
- Plots loss curves and saves `main.keras`

### Step 4: Convert to TensorFlow Lite

```bash
python convert_model.py
```

- Converts `.keras` to `.tflite`  
- Saves final model as `converted_model.tflite`

---

## 📉 Model Performance

- **NRMSD**: ~2% on the test set  
- Model captures trends over a 168-hour window with high precision  
- Inference is lightweight and suitable for CPU-only deployment

---

## 🧪 Features Used

- `temp`: air temperature (target)
- `dwpt`: dew point
- `rhum`: relative humidity
- `prcp`: precipitation
- `wdir`: wind direction
- `wspd`: wind speed
- `pres`: pressure
- `coco`: weather condition code

---

## 📊 Dependencies

- `tensorflow`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `meteostat`

---

## 📝 Acknowledgments

- Meteostat API for free access to historical weather data  
- TensorFlow & Keras for model building  
- STM32Cube.AI for ML-on-MCU deployment tools

MIT License – see `LICENSE` for details.

---
