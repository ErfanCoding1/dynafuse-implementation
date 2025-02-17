import pickle
import numpy as np
from scipy.signal import resample
import pandas as pd

# Load data from pickle file
def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data

# Extract sensor data
def extract_sensor_data(data):
    ppg_data = data['signal']['wrist']['BVP']
    eda_data = data['signal']['wrist']['EDA']
    ecg_data = data['signal']['chest']['ECG']
    return ppg_data, eda_data, ecg_data

# Synchronize sensor data based on shortest length
def synchronize_data(ppg, eda, ecg):
    min_length = min(len(ppg), len(eda), len(ecg))
    ppg_sync = resample(ppg, min_length)
    eda_sync = resample(eda, min_length)
    ecg_sync = resample(ecg, min_length)
    return ppg_sync, eda_sync, ecg_sync

# Detect stress based on simple thresholding
def detect_stress(ppg, eda, ecg):
    stress_labels = []
    for i in range(len(ppg)):
        ppg_feature = ppg[i][0]
        eda_feature = eda[i][0]
        ecg_feature = ecg[i][0]
        
        if (ppg_feature > 50 or ppg_feature < -50) and (eda_feature > 0.6 or eda_feature < 0.38) and (ecg_feature > 0.05 or ecg_feature < -0.05):
            stress_labels.append(1)
        else:
            stress_labels.append(0)
    return np.array(stress_labels)

# Split data into batches
def create_batches(data, batch_size=100):
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

# Calculate energy consumption for each batch
def calculate_energy(batch, power_coefficient):
    try:
        batch_array = np.array(batch)
        values = batch_array[:, 0]
        return np.sum(np.abs(values)) * power_coefficient
    except Exception as e:
        print(f"Error calculating energy for batch: {batch}, Error: {e}")
        return 0

# Execute pipeline
data = load_data("with_my_dataset/S8.pkl")
ppg, eda, ecg = extract_sensor_data(data)
ppg_sync, eda_sync, ecg_sync = synchronize_data(ppg, eda, ecg)
stress_labels = detect_stress(ppg_sync, eda_sync, ecg_sync)

ppg_batches = create_batches(ppg_sync, batch_size=100)
eda_batches = create_batches(eda_sync, batch_size=100)
ecq_batches = create_batches(ecg_sync, batch_size=100)
batch_labels = [1 if np.mean(batch) > 0.3 else 0 for batch in create_batches(stress_labels, 100)]

energy_ppg = [calculate_energy(batch, 0.001) for batch in ppg_batches]
energy_eda = [calculate_energy(batch, 0.05) for batch in eda_batches]
energy_ecg = [calculate_energy(batch, 0.2) for batch in ecq_batches]

batch_data = []
for i in range(len(ppg_batches)):
    batch_data.append({
        "PPG_Batch": str(np.array(ppg_batches[i]).tolist()),
        "EDA_Batch": str(np.array(eda_batches[i]).tolist()),
        "ECG_Batch": str(np.array(ecq_batches[i]).tolist()),
        "Stress_Label": batch_labels[i],
        "Energy_PPG": energy_ppg[i],
        "Energy_EDA": energy_eda[i],
        "Energy_ECG": energy_ecg[i]
    })

batch_df = pd.DataFrame(batch_data)
batch_df.to_csv("with_my_dataset/sensor_batches.csv", index=False)

print(batch_df.head(10))
