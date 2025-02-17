import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the noisy sensor batches data
    df = pd.read_csv("with_my_dataset/sensor_batches_noisy.csv")
    
    # Ensure energy columns are numeric
    df['Energy_PPG'] = df['Energy_PPG'].astype(float)
    df['Energy_EDA'] = df['Energy_EDA'].astype(float)
    df['Energy_ECG'] = df['Energy_ECG'].astype(float)
    
    # Compute the average energy consumption per batch
    df['Average_Energy'] = df[['Energy_PPG', 'Energy_EDA', 'Energy_ECG']].mean(axis=1)
    
    # Normalize energy consumption to [0, 1] by dividing by the maximum average energy
    max_energy = df['Average_Energy'].max()
    df['Normalized_Energy'] = df['Average_Energy'] / max_energy
    
    # Create a Batch_Index column (each row corresponds to one batch)
    df['Batch_Index'] = np.arange(1, len(df) + 1)
    
    # Plot the normalized energy consumption with a horizontal baseline at 1
    plt.figure(figsize=(4.64, 3.22))  # Matching previous figure dimensions
    plt.plot(df['Batch_Index'], df['Normalized_Energy'], marker='o', linestyle='-', color='blue', label="Normalized Energy")
    plt.axhline(y=1.0, color="red", linestyle="--", label="Baseline = 1")
    plt.xlabel("Batch Index")
    plt.ylabel("Normalized Energy Consumption")
    plt.title("Normalized Energy Consumption per Batch")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.savefig("with_my_dataset/energy_comparison_normalized.png")
    plt.show()
    
    # Save the normalized energy data to CSV
    df[['Batch_Index', 'Normalized_Energy']].to_csv("with_my_dataset/energy_comparison_normalized.csv", index=False)
    print("Normalized energy comparison data saved to with_my_dataset/energy_comparison_normalized.csv")

if __name__ == "__main__":
    main()
