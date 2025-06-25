import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Retrieve the temperature data from the csv
data = pd.read_csv('temperature_data.csv')
data.sort_values(by='time', inplace=True)
data.reset_index(drop=True, inplace=True)

# Choose the columns for which we have numerical data and make sure that any faulty data is converted to NaN
select_columns = ['temp','dwpt','rhum','prcp','wdir','wspd','pres','coco']
feature_data = data[select_columns].apply(pd.to_numeric, errors='coerce')



# Replace all NaNs with the corresponding mean where possible
feature_data = feature_data.fillna(feature_data.mean())
# feature_data = feature_data.sample(frac=1)
print(feature_data.isna().sum())

# The portion of data to use for training
SPLIT = 0.8

# The number of rows in each type of data
train_size = int(len(feature_data) * SPLIT)
val_size = int(len(feature_data) * (1-SPLIT)//2)
test_size = len(feature_data) - train_size - val_size

# Defining the actual datasets
data_train = feature_data[:train_size]
data_val = feature_data[train_size:train_size + val_size]
data_test = feature_data[-test_size:]

''' Display the data split
data_train['temp'].plot(legend=True)
data_val['temp'].plot(legend=True)
data_test['temp'].plot(legend=True)
plt.legend(['Train', 'Val', 'Test'])
plt.suptitle('Temperature')
plt.ylabel('T (degC)')
'''

# Normalizing the datasets using the MinMaxScaler from scikit-learn
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_normalized = scaler.fit_transform(data_train)
data_val_normalized = scaler.transform(data_val)
data_test_normalized = scaler.transform(data_test)

# Save the scaler parameters to a text file
with open('scaler_params.txt', 'w') as file:
    file.write('scale:' + ','.join(map(str, scaler.scale_)) + '\n')
    file.write('min:' + ','.join(map(str, scaler.min_)) + '\n')
    file.write('data_min:' + ','.join(map(str, scaler.data_min_)) + '\n')
    file.write('data_max:' + ','.join(map(str, scaler.data_max_)) + '\n')
    file.write('data_range:' + ','.join(map(str, scaler.data_range_)) + '\n')

print(f'Train: {data_train_normalized.min()}, {data_train_normalized.max()}')
print(f'Test:  {data_test_normalized.min()}, {data_test_normalized.max()}')
print(f'Val:   {data_val_normalized.min()}, {data_val_normalized.max()}')