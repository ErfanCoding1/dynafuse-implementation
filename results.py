import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy data for each batch
df = pd.read_csv("with_my_dataset/per_batch_accuracies.csv")

# Compute weights for each sensor per batch
df["Total_Accuracy"] = df["Accuracy_PPG"] + df["Accuracy_EDA"] + df["Accuracy_ECG"]
df["Weight_PPG"] = df["Accuracy_PPG"] / df["Total_Accuracy"]
df["Weight_EDA"] = df["Accuracy_EDA"] / df["Total_Accuracy"]
df["Weight_ECG"] = df["Accuracy_ECG"] / df["Total_Accuracy"]

sensors = ["PPG", "EDA", "ECG"]
titles = ["(I) PPG", "(II) EDA", "(III) ECG"]
weight_cols = {"PPG": "Weight_PPG", "EDA": "Weight_EDA", "ECG": "Weight_ECG"}

# Create figure with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

for i, sensor in enumerate(sensors):
    axs[i].scatter(df["Batch_Index"], df[weight_cols[sensor]], marker="x", color="blue", label=f"{sensor} Weight")
    axs[i].hlines(0.3, xmin=df["Batch_Index"].min(), xmax=df["Batch_Index"].max(), colors="red", linestyles="solid")
    axs[i].set_xlabel("Batch number", fontsize=10)
    axs[i].set_title(titles[i], fontsize=12)
    axs[i].grid(False)  # Remove background grid for better visualization

axs[0].set_ylabel("Weight", fontsize=10)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3)

plt.tight_layout()
plt.savefig("with_my_dataset/sensor_weights_adjusted.png", dpi=300)
plt.show()
