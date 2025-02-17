import pandas as pd
import numpy as np
import ast
import re

# Load data from CSV file
def load_data(file_path):
    return pd.read_csv(file_path, dtype=str)

# Convert batch string to a flattened list of floats
def parse_batch(batch_str):
    if pd.isnull(batch_str):
        return []
    try:
        parsed = ast.literal_eval(batch_str)
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], list):
                return [float(item[0]) for item in parsed if isinstance(item, list) and len(item) > 0]
            else:
                return [float(x) for x in parsed]
        else:
            return []
    except Exception as e:
        print(f"Error parsing batch: {batch_str}, Error: {e}")
        return []

# Add relative noise to each batch
def add_noise_to_batch_relative(batch, noise_ratio=0.05):
    values = parse_batch(batch)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return str([])
    std_signal = np.std(arr)
    noise = np.random.normal(0, noise_ratio * std_signal, arr.shape)
    noisy_values = arr + noise
    noisy_list = [[float(x)] for x in noisy_values]
    return str(noisy_list)

# Apply noise to sensor data
def add_noise(data, noise_ratios={'PPG_Batch': 0.09, 'EDA_Batch': 0.05, 'ECG_Batch': 0.003}):
    noisy_data = data.copy()
    for col in ['PPG_Batch', 'EDA_Batch', 'ECG_Batch']:
        ratio = noise_ratios.get(col, 0.05)
        noisy_data[col] = noisy_data[col].apply(lambda x: add_noise_to_batch_relative(x, noise_ratio=ratio))
    return noisy_data

# Calculate SNR for a batch
def calculate_snr(batch, noise_ratio):
    values = parse_batch(batch)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0
    signal_power = np.mean(np.square(arr))
    noise_power = (noise_ratio ** 2) * signal_power
    return 10 * np.log10(signal_power / (noise_power + 1e-10))

# Compute energy consumption for a batch
def calculate_energy(batch, power_coefficient):
    values = parse_batch(batch)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0
    return np.sum(np.abs(arr)) * power_coefficient

def main():
    # Load the data from CSV file
    data = load_data("with_my_dataset/sensor_batches.csv")
    
    # Define noise ratios for each sensor
    noise_ratios = {'PPG_Batch': 0.1, 'EDA_Batch': 0.05, 'ECG_Batch': 0.05}
    data_noisy = add_noise(data, noise_ratios=noise_ratios)
    
    # Calculate SNR for each sensor
    data_noisy['SNR_PPG'] = data_noisy['PPG_Batch'].apply(lambda x: calculate_snr(x, noise_ratio=0.1))
    data_noisy['SNR_EDA'] = data_noisy['EDA_Batch'].apply(lambda x: calculate_snr(x, noise_ratio=0.05))
    data_noisy['SNR_ECG'] = data_noisy['ECG_Batch'].apply(lambda x: calculate_snr(x, noise_ratio=0.05))
    
    # Calculate energy consumption for each sensor
    data_noisy['Energy_PPG'] = data_noisy['PPG_Batch'].apply(lambda x: calculate_energy(x, 0.001))
    data_noisy['Energy_EDA'] = data_noisy['EDA_Batch'].apply(lambda x: calculate_energy(x, 0.05))
    data_noisy['Energy_ECG'] = data_noisy['ECG_Batch'].apply(lambda x: calculate_energy(x, 0.2))
    
    # Save the noisy data to a new CSV file
    data_noisy.to_csv("with_my_dataset/sensor_batches_noisy.csv", index=False)
    print(data_noisy.head())

if __name__ == "__main__":
    main()
