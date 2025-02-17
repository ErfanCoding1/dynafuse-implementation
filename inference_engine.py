import os
import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

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
    """Convert the sensor batch string into a numpy array of shape (batch_size, 1)."""
    values = parse_batch(batch_str)
    if len(values) < batch_size:
        values = values + [0] * (batch_size - len(values))
    else:
        values = values[:batch_size]
    return np.array(values).reshape(-1, 1)

def main():
    # Load sensor batch data and model weights
    sensor_data = pd.read_csv("with_my_dataset/sensor_batches_noisy.csv", dtype=str)
    selected_df = pd.read_csv("with_my_dataset/selected_models_weights.csv")
    
    # Build a dictionary mapping sensor names to their final weight
    selected_models = {}
    for idx, row in selected_df.iterrows():
        sensor = row["Sensor_Used"].replace("_Batch", "")
        weight = row["Final_Weight"]
        if weight > 0:
            selected_models[sensor] = weight
    
    if len(selected_models) == 1:
        for sensor in selected_models:
            selected_models[sensor] = 1.0

    # Load models for selected sensors
    model_dict = {}
    for sensor in selected_models.keys():
        model_path = os.path.join("with_my_dataset/saved_models", f"model_{sensor}.h5")
        if os.path.exists(model_path):
            model_dict[sensor] = load_model(model_path)
        else:
            print(f"Model file for sensor {sensor} not found at {model_path}")
    
    # Iterate over each batch for inference
    predictions = []
    batch_indices = []
    for idx, row in sensor_data.iterrows():
        sensor_preds = {}
        for sensor in selected_models.keys():
            col_name = f"{sensor}_Batch"
            if col_name in row:
                input_data = prepare_model_input(row[col_name])
                input_data = np.expand_dims(input_data, axis=0)  # shape: (1, batch_size, 1)
                pred_prob = model_dict[sensor].predict(input_data, verbose=0)
                avg_pred = np.mean(pred_prob)
                sensor_preds[sensor] = avg_pred
            else:
                print(f"Column {col_name} not found in row {idx}")
        
        if len(sensor_preds) == 0:
            continue
        
        if len(sensor_preds) == 1:
            final_prob = list(sensor_preds.values())[0]
        else:
            # Weighted average of predictions using final weights
            weighted_sum = sum(selected_models[sensor] * prob for sensor, prob in sensor_preds.items())
            total_weight = sum(selected_models.values())
            final_prob = weighted_sum / total_weight if total_weight != 0 else 0
        
        # Apply threshold for stress prediction
        final_pred = 1 if final_prob >= 0.3 else 0
        predictions.append(final_pred)
        batch_indices.append(idx + 1)
    
    # Save inference results to CSV
    results_df = pd.DataFrame({
        "Batch_Index": batch_indices,
        "Predicted_Stress": predictions
    })
    results_df.to_csv("with_my_dataset/inference_results.csv", index=False)
    print("Inference results saved to with_my_dataset/inference_results.csv")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df["Batch_Index"], results_df["Predicted_Stress"], marker="o", color="blue", s=100, label="Predicted Stress")
    plt.xlabel("Batch Index")
    plt.ylabel("Predicted Stress (0: No, 1: Yes)")
    plt.title("Inference Results per Batch")
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.axhline(y=0.5, color="red", linestyle="--", label="Threshold = 0.5")
    plt.legend()
    plt.savefig("with_my_dataset/inference_results.png")
    plt.show()

if __name__ == "__main__":
    main()
 