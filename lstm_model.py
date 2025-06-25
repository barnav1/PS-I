import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Retrieve the temperature data from the csv
data = pd.read_csv("temperature_data.csv")
data.sort_values(by="time", inplace=True)
data.reset_index(drop=True, inplace=True)

# Choose the columns for which we have numerical data and make sure that any faulty data is converted to NaN
select_columns = ["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "pres", "coco"]
feature_data = data[select_columns].apply(pd.to_numeric, errors="coerce")


# Replace all NaNs with the corresponding mean where possible
feature_data = feature_data.fillna(feature_data.mean())
# feature_data = feature_data.sample(frac=1)

# The portion of data to use for training
SPLIT = 0.8

# The number of rows in each type of data
train_size = int(len(feature_data) * SPLIT)
val_size = int(len(feature_data) * (1 - SPLIT) // 2)
test_size = len(feature_data) - train_size - val_size

# Defining the actual datasets
data_train = feature_data[:train_size]
data_val = feature_data[train_size : train_size + val_size]
data_test = feature_data[-test_size:]


# Normalizing the datasets using the MinMaxScaler from scikit-learn
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_normalized = scaler.fit_transform(data_train)
data_val_normalized = scaler.transform(data_val)
data_test_normalized = scaler.transform(data_test)

# Defining some parameters for the sequence generating function
model_output = "temp"
steps = 168
input_num = len(feature_data.columns)
model_output_index = select_columns.index(model_output)


def sequence(input_data, n_steps, output_index):
    X, y = [], []
    for i in range(len(input_data) - n_steps - 1):
        # The end of the input sequence
        end_idx = i + n_steps
        # The end of the output sequence
        out_end_idx = end_idx + 1
        # Check that the index isn't out of bounds
        if out_end_idx > len(input_data):
            break
        # Retrieve input and output
        seq_x, seq_y = (
            input_data[i:end_idx, :],
            input_data[out_end_idx - 1, output_index],
        )
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Creating the training, validation, and testing data for the model
X_train, y_train = sequence(data_train_normalized, steps, model_output_index)
X_val, y_val = sequence(data_val_normalized, steps, model_output_index)
X_test, y_test = sequence(data_test_normalized, steps, model_output_index)

# Creating the LSTM model with 1 dense layer for output
model = Sequential([LSTM(128, input_shape=(steps, X_train.shape[2])), Dense(1)])

model.compile(optimizer="adam", loss="mse")

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, mode="min", restore_best_weights=True
)


BATCH = 32
EPOCHS = 20

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Training History")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
