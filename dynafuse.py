import os
import pandas as pd
import numpy as np

# Input file path (output of model training stage)
MODEL_RESULTS_FILE = "with_my_dataset/model_pool_results.csv"
# Output file path (to store selected models and weights)
OUTPUT_FILE = "with_my_dataset/selected_models_weights.csv"

# Operation mode: one of "high_accuracy", "nominal", "energy_saving"
mode = "energy_saving"

# Load trained model results
df = pd.read_csv(MODEL_RESULTS_FILE)

# Ensure Validation_Accuracy is float
df['Validation_Accuracy'] = df['Validation_Accuracy'].astype(float)

# Select the best model for each sensor based on highest accuracy
best_models = df.sort_values("Validation_Accuracy", ascending=False).groupby("Sensor_Used", as_index=False).first()

# Compute initial weight as accuracy divided by total accuracy
total_accuracy = best_models['Validation_Accuracy'].sum()
best_models['Initial_Weight'] = best_models['Validation_Accuracy'] / total_accuracy

# Define minimum accuracy threshold: 0.8 * max accuracy
max_accuracy = best_models['Validation_Accuracy'].max()
min_accuracy_threshold = 0.8 * max_accuracy

print("Best Models per Sensor (Before Energy Demand Selection):")
print(best_models[['Sensor_Used', 'Validation_Accuracy', 'Initial_Weight']])
print(f"Maximum Accuracy = {max_accuracy:.4f}, Minimum Accuracy Threshold = {min_accuracy_threshold:.4f}\n")

# Function to select models based on operation mode
def select_models(df, mode, min_acc_threshold):
    """
    Args:
      df: DataFrame containing unimodal models (one row per sensor)
      mode: Operation mode, one of "high_accuracy", "nominal", "energy_saving"
      min_acc_threshold: Minimum accuracy threshold (0.8 * max_accuracy)
    Returns:
      A DataFrame with a 'Selected' column indicating model selection (1 for selected, 0 for not selected)
    """
    df = df.copy()
    df['Selected'] = 0

    if mode == "high_accuracy":
        max_val = df['Validation_Accuracy'].max()
        idx_max = df.index[df['Validation_Accuracy'] == max_val]
        df.loc[idx_max, 'Selected'] = 1

    elif mode == "nominal":
        candidate_mask = df['Validation_Accuracy'] >= min_acc_threshold
        candidates = df[candidate_mask]
        if candidates.empty:
            print("No unimodal model meets the minimum accuracy criteria.")
        else:
            k = max(1, int(np.floor(0.7 * len(candidates))))
            candidates = candidates.sort_values("Average_Energy", ascending=True)
            selected_indices = candidates.head(k).index
            df.loc[selected_indices, 'Selected'] = 1

    elif mode == "energy_saving":
        candidate_mask = df['Validation_Accuracy'] >= min_acc_threshold
        candidates = df[candidate_mask]
        if candidates.empty:
            print("No unimodal model meets the minimum accuracy criteria.")
        else:
            min_energy = candidates['Average_Energy'].min()
            idx_min = candidates.index[candidates['Average_Energy'] == min_energy]
            df.loc[idx_min, 'Selected'] = 1
    else:
        raise ValueError("Invalid mode. Choose one of: high_accuracy, nominal, energy_saving.")
    
    return df

# Apply model selection based on chosen mode
selected_df = select_models(best_models, mode, min_accuracy_threshold)

# Compute final weights: Initial weight multiplied by Selected flag
selected_df['Final_Weight_Raw'] = selected_df['Initial_Weight'] * selected_df['Selected']
sum_final = selected_df['Final_Weight_Raw'].sum()
if sum_final > 0:
    selected_df['Final_Weight'] = selected_df['Final_Weight_Raw'] / sum_final
else:
    selected_df['Final_Weight'] = 0

print("Selected Models and Final Weights:")
print(selected_df[['Sensor_Used', 'Validation_Accuracy', 'Average_Energy', 'Initial_Weight', 'Selected', 'Final_Weight']])

# Save results to CSV for next processing stage
selected_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nResults saved to {OUTPUT_FILE}.")
