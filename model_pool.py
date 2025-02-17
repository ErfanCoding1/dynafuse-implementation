import os
import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

# Function to parse batch string data into a list
def parse_batch(batch_str):
    try:
        parsed = ast.literal_eval(batch_str)
        if isinstance(parsed, list):
            return parsed
        else:
            return []
    except Exception as e:
        print(f"Error parsing batch: {batch_str}, Error: {e}")
        return []

# Load dataset
data = pd.read_csv("with_my_dataset/sensor_batches_noisy.csv")

# Convert sensor columns into list format
for col in ['PPG_Batch', 'EDA_Batch', 'ECG_Batch']:
    data[col] = data[col].apply(parse_batch)

data['Stress_Label'] = data['Stress_Label'].astype(int)

# Define different sensor combinations
sensor_combinations = {
    "PPG": {"input_cols": ["PPG_Batch"], "energy_cols": ["Energy_PPG"]},
    "EDA": {"input_cols": ["EDA_Batch"], "energy_cols": ["Energy_EDA"]},
    "ECG": {"input_cols": ["ECG_Batch"], "energy_cols": ["Energy_ECG"]}
}

# Function to prepare input data for training and testing
def prepare_data(data, input_cols, batch_size=100):
    X_list, y_list = [], []
    for _, row in data.iterrows():
        batch_channels = []
        valid = True
        for col in input_cols:
            lst = row[col]
            if isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], list):
                lst = [x[0] for x in lst]
            if not isinstance(lst, list):
                valid = False
                break
            if len(lst) < batch_size:
                lst = lst + [0] * (batch_size - len(lst))
            elif len(lst) > batch_size:
                lst = lst[:batch_size]
            batch_channels.append(lst)
        if valid:
            X_sample = np.array(batch_channels).T  # Shape: (batch_size, num_channels)
            X_list.append(X_sample)
            y_list.append(row["Stress_Label"])
    return np.array(X_list), np.array(y_list)

# Function to define the neural network model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train models for each sensor and save results
results = []
if not os.path.exists("with_my_dataset/saved_models"):
    os.makedirs("with_my_dataset/saved_models")

for model_idx, (model_name, params) in enumerate(sensor_combinations.items()):
    print(f"Training model for sensor: {model_name}")
    input_cols, energy_cols = params["input_cols"], params["energy_cols"]
    X, y = prepare_data(data, input_cols)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42 + model_idx)
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    
    epochs = 20  # Number of training epochs for single-sensor models
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(8)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(8)
    
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=0)
    loss, accuracy = model.evaluate(val_dataset, verbose=0)
    
    # Compute average energy consumption if energy columns exist
    energy = data[energy_cols].sum(axis=1).mean() if set(energy_cols).issubset(data.columns) else None
    
    model_path = os.path.join("with_my_dataset/saved_models", f"model_{model_name}.h5")
    model.save(model_path)
    
    results.append({
        "Model_Name": model_name,
        "Sensor_Used": input_cols[0],
        "Validation_Accuracy": accuracy,
        "Average_Energy": energy,
        "Model_Path": model_path
    })
    print(f"Model: {model_name}, Accuracy: {accuracy:.4f}, Energy: {energy if energy is not None else 'N/A'}, Saved at: {model_path}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("with_my_dataset/model_pool_results.csv", index=False)
print("\nModel Pool Results:")
print(results_df)
