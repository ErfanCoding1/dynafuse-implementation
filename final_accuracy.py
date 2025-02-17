import os
import pandas as pd
import numpy as np
import ast
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- Utility Functions ---
def parse_batch(batch_str):
    """Convert a batch string to a flattened list of floats."""
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

def prepare_model_input(batch_str, batch_size=100):
    """
    Convert the sensor batch string into a numpy array of shape (batch_size, 1).
    Pads with zeros if necessary.
    """
    values = parse_batch(batch_str)
    if len(values) < batch_size:
        values = values + [0] * (batch_size - len(values))
    else:
        values = values[:batch_size]
    return np.array(values).reshape(-1, 1)

# --- Inference and Per-Batch Accuracy Computation ---
def compute_batch_accuracy(model_dict, selected_weights, sensor_data, true_labels, batch_size=100):
    """
    For each batch, compute the weighted average predicted probability across the selected sensors.
    Then compute a continuous accuracy measure: accuracy = (1 - |avg_prob - true_label|) * 100.
    Returns a list of accuracy percentages per batch.
    """
    batch_accuracies = []
    n_batches = len(true_labels)
    for idx in range(n_batches):
        sensor_preds = {}
        for sensor, weight in selected_weights.items():
            col_name = f"{sensor}_Batch"
            if col_name in sensor_data.columns:
                batch_str = sensor_data.loc[idx, col_name]
                input_data = prepare_model_input(batch_str, batch_size=batch_size)
                input_data = np.expand_dims(input_data, axis=0)  # shape: (1, batch_size, 1)
                pred_prob = model_dict[sensor].predict(input_data, verbose=0).flatten()
                avg_prob = np.mean(pred_prob)
                sensor_preds[sensor] = avg_prob
            else:
                print(f"Column {col_name} not found in row {idx}")
        if len(sensor_preds) == 0:
            batch_accuracies.append(0)
            continue
        # If only one sensor is selected, use its prediction; if more, compute weighted average.
        if len(sensor_preds) == 1:
            final_prob = list(sensor_preds.values())[0]
        else:
            weighted_sum = sum(selected_weights[sensor] * sensor_preds[sensor] for sensor in sensor_preds)
            total_weight = sum(selected_weights[sensor] for sensor in sensor_preds)
            final_prob = weighted_sum / total_weight if total_weight != 0 else 0
        error = abs(final_prob - true_labels[idx])
        batch_acc = (1 - error) * 100
        batch_accuracies.append(batch_acc)
    return batch_accuracies

def main():
    # Load the noisy sensor data with ground truth labels
    sensor_data = pd.read_csv("with_my_dataset/sensor_batches_noisy.csv", dtype=str)
    # Convert ground truth Stress_Label to int (assumed binary 0/1)
    sensor_data['Stress_Label'] = sensor_data['Stress_Label'].astype(int)
    
    # Load the selection file with final sensor weights
    selected_df = pd.read_csv("with_my_dataset/selected_models_weights.csv")
    # Build a dictionary mapping sensor (e.g., "PPG") to its final weight.
    selected_weights = {}
    for idx, row in selected_df.iterrows():
        sensor = row["Sensor_Used"]
        if sensor.endswith("_Batch"):
            sensor = sensor.replace("_Batch", "")
        weight = float(row["Final_Weight"])
        if weight > 0:
            selected_weights[sensor] = weight
    if len(selected_weights) == 1:
        # If only one sensor is selected, set its weight to 1.
        for sensor in selected_weights:
            selected_weights[sensor] = 1.0
    print("Selected sensor weights:", selected_weights)
    
    # Load corresponding models for the selected sensors.
    model_dict = {}
    for sensor in selected_weights.keys():
        model_path = os.path.join("with_my_dataset/saved_models", f"model_{sensor}.h5")
        if os.path.exists(model_path):
            model_dict[sensor] = load_model(model_path)
        else:
            print(f"Model file for sensor {sensor} not found at {model_path}")
    
    # Get ground truth labels per batch (assume one row per batch)
    true_labels = sensor_data['Stress_Label'].tolist()
    
    # Compute per-batch accuracy using the final fused prediction.
    batch_accuracies = compute_batch_accuracy(model_dict, selected_weights, sensor_data, true_labels, batch_size=100)
    
    # Save per-batch accuracy results to CSV.
    accuracy_df = pd.DataFrame({
        "Batch_Index": list(range(1, len(true_labels) + 1)),
        "Final_Accuracy": batch_accuracies
    })
    accuracy_df.to_csv("with_my_dataset/per_batch_final_accuracy.csv", index=False)
    print("Per-batch final accuracy saved to with_my_dataset/per_batch_final_accuracy.csv")
    
    # Plot the per-batch accuracy chart.
    plt.figure(figsize=(4.64, 3.22))
    plt.plot(accuracy_df["Batch_Index"], accuracy_df["Final_Accuracy"], marker="o", linestyle="-", color="blue")
    plt.xlabel("Batch Number")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Batch Final Accuracy")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig("with_my_dataset/per_batch_accuracy.png")
    plt.show()

if __name__ == "__main__":
    main()
