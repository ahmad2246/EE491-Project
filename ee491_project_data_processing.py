"""# Import Necessary Packages"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pywt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

"""## Loading Dataset"""

dataset_path = "/content/Complete Wind Data.xls"
data = pd.read_excel(dataset_path)
print(f"data.shape = {data.shape}")
data.head()

"""## Extracting information of signle site (A)

"""

site_a_columns = [
    'Time',
    'WTGCA - Ambient.Temperature.Average',
    'WTGCA - Ambient.WinDir.Absolute.Average',
    'WTGCA - Ambient.WindSpeed.Average',
    'WTGCA - Ambient.WindSpeed.Maximum',
    'WTGCA - Ambient.WindSpeed.Minimum',
    'WTGCA - Grid.Production.Power.Average',
    'WTGCA - Grid.Production.Power.Maximum',
    'WTGCA - Grid.Production.Power.Minimum'
]

data = data[site_a_columns]
print(f"data.shape = {data.shape}")
data.head()

"""# Extract meaningful features from the "Time" column

"""

data['Time'] = pd.to_datetime(data['Time'], infer_datetime_format=True)
data['Month'] = data['Time'].dt.month
data['Day'] = data['Time'].dt.day
data['Hour'] = data['Time'].dt.hour

"""# Handling Categorical Variables"""

# Circular encoding for time features
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)

# Drop the original time-related features
data = data.drop(['Month', 'Day', 'Hour'], axis=1)

"""# Visualizing Dataset of Site A

"""

sns.set(style="whitegrid")

data_columns = [
    'WTGCA - Ambient.Temperature.Average',
    'WTGCA - Ambient.WinDir.Absolute.Average',
    'WTGCA - Ambient.WindSpeed.Average',
    'WTGCA - Ambient.WindSpeed.Maximum',
    'WTGCA - Ambient.WindSpeed.Minimum',
    'WTGCA - Grid.Production.Power.Average',
    'WTGCA - Grid.Production.Power.Maximum',
    'WTGCA - Grid.Production.Power.Minimum'
]

titles = [
    'Temperature Variation Over Time for Site A',
    'Wind Direction Variation Over Time for Site A',
    'Average Wind Speed Variation Over Time for Site A',
    'Maximum Wind Speed Variation Over Time for Site A',
    'Minimum Wind Speed Variation Over Time for Site A',
    'Average Power Variation Over Time for Site A',
    'Maximum Power Variation Over Time for Site A',
    'Minimum Power Variation Over Time for Site A'
]

y_labels = [
    'Temperature (°C)',
    'Wind Direction',
    'Wind Speed (m/s)',
    'Wind Speed (m/s)',
    'Wind Speed (m/s)',
    'Power (kW)',
    'Power (kW)',
    'Power (kW)'
]


# Filter data for 6 months
start_date = '1/1/2006  12:00:00 AM'
end_date = '1/7/2006  12:00:00 AM'
data_filtered = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)]


fig, axes = plt.subplots(4, 2, figsize=(10, 16))

for i, column in enumerate(data_columns):
    row, col = divmod(i, 2)
    sns.lineplot(ax=axes[row, col], x='Time', y=column, data=data_filtered)
    axes[row, col].set_title(titles[i])
    axes[row, col].set_xlabel('Time')
    axes[row, col].set_ylabel(y_labels[i])
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

"""# Handling Missing Values"""

print(data.info())  # Before dropping
data.dropna(inplace=True)  # Drop empty rows
print(data.info())  # After dropping

"""# Removing Outliers

"""

numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
z_scores = zscore(data[numerical_columns])
threshold = 3
outliers_mask = (abs(z_scores) > threshold).any(axis=1)
outliers = data[outliers_mask]
print("Number of outliers:", len(outliers))
data_no_outliers = data[~outliers_mask]

"""# Applying Wavelet transform for signal processing and denoising

"""

# Applying Wavelet transform for signal processing and denoising
def wavelet_denoise(data, feature, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data[feature], wavelet, level=level)
    threshold = 0.1
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised_data = pywt.waverec(coeffs, wavelet)

    # Ensure the lengths match
    min_length = min(len(data), len(denoised_data))
    denoised_data = denoised_data[:min_length]

    return denoised_data

sample_feature = 'WTGCA - Ambient.Temperature.Average'
data_no_outliers_denoised = data_no_outliers.copy()

# Align index before assigning
data_no_outliers_denoised[sample_feature] = wavelet_denoise(data_no_outliers, sample_feature)

plt.figure(figsize=(12, 6))
sns.lineplot(x='Time', y=sample_feature, data=data_no_outliers, label='Original Data')
sns.lineplot(x='Time', y=sample_feature, data=data_no_outliers_denoised, label='Denoised Data')
plt.title(f'Effect of Wavelet Denoising on {sample_feature}')
plt.xlabel('Time')
plt.ylabel(sample_feature)
plt.legend()
plt.xticks(rotation=45)
plt.show()

"""# Visualizing Dataset of Site A After:
- Removing missing values
- Handling outliers
- Applying signal processing
"""

# Filter data for 6 months
start_date = '1/1/2006  12:00:00 AM'
end_date = '1/7/2006  12:00:00 AM'
data_filtered = data_no_outliers_denoised[(data_no_outliers_denoised['Time'] >= start_date) & (data_no_outliers_denoised['Time'] <= end_date)]

fig, axes = plt.subplots(4, 2, figsize=(10, 16))

for i, column in enumerate(data_columns):
    row, col = divmod(i, 2)
    sns.lineplot(ax=axes[row, col], x='Time', y=column, data=data_filtered)
    axes[row, col].set_title(titles[i])
    axes[row, col].set_xlabel('Time')
    axes[row, col].set_ylabel(y_labels[i])
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

"""# Normalization"""

features_to_normalize = data_no_outliers_denoised.drop(['Time'], axis=1)
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(features_to_normalize), columns=features_to_normalize.columns)

data_normalized['Time'] = data_no_outliers_denoised['Time']

"""# Visualizing Dataset of Site A After:
- Removing missing values
- Handling outliers
- Applying signal processing
- Norrmalization
"""

# Filter data for 6 months
start_date = '1/1/2006  12:00:00 AM'
end_date = '1/7/2006  12:00:00 AM'
data_filtered = data_normalized[(data_normalized['Time'] >= start_date) & (data_normalized['Time'] <= end_date)]

fig, axes = plt.subplots(4, 2, figsize=(10, 16))

for i, column in enumerate(data_columns):
    row, col = divmod(i, 2)
    sns.lineplot(ax=axes[row, col], x='Time', y=column, data=data_filtered)
    axes[row, col].set_title(titles[i])
    axes[row, col].set_xlabel('Time')
    axes[row, col].set_ylabel(y_labels[i])
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

