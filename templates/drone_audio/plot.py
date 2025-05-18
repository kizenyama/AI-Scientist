import json
import os
import os.path as osp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
results_info = {}

# Process all run directories (even though we expect only run_0 for this experiment)
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        # Load the all_results.json file
        with open(osp.join(folder, "all_results.json"), "r") as f:
            all_results = json.load(f)
        
        # Extract training history for each model
        run_info = {}
        for model_type, results in all_results.items():
            run_info[model_type] = {
                "steps": results["steps"],
                "train_accuracy": results["history"]["accuracy"],
                "val_accuracy": results["history"]["val_accuracy"],
                "test_accuracy": results["test_accuracy"],
                "model_size_kb": results["model_size_kb"],
                "quantized_size_kb": results["quantized_size_kb"]
            }
        
        results_info[folder] = run_info
        
        # Also load final_info.json if it exists (for AI Scientist compatibility)
        final_info_path = osp.join(folder, "final_info.json")
        if osp.exists(final_info_path):
            with open(final_info_path, "r") as f:
                final_results[folder] = json.load(f)

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Binary Drone Classification",
}

# Create a programmatic color palette for models
def generate_color_palette(n):
    cmap = plt.get_cmap('tab10')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 0.5, n)]

# Get the list of runs
runs = list(labels.keys())

# Model types and their colors
model_types = ["cnn", "lightweight_cnn", "dscnn"]
model_labels = {
    "cnn": "CNN",
    "lightweight_cnn": "Lightweight CNN", 
    "dscnn": "DS-CNN"
}
model_colors = generate_color_palette(len(model_types))

# Since we have one dataset, we'll save with consistent naming
dataset_name = "drone_classification"

# Plot 1: Validation Accuracy Over Training
plt.figure(figsize=(10, 6))

# Assuming we're using run_0 data (since that's the baseline)
run = "run_0"
for i, model_type in enumerate(model_types):
    steps = results_info[run][model_type]["steps"]
    val_accuracy = results_info[run][model_type]["val_accuracy"]
    
    plt.plot(steps, val_accuracy, label=model_labels[model_type], 
            color=model_colors[i], linewidth=2)

plt.title(f"Validation Accuracy Over Training - {dataset_name.replace('_', ' ').title()}")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"val_accuracy_{dataset_name}.png", dpi=300)
plt.close()

# Plot 2: Training Accuracy Over Training  
plt.figure(figsize=(10, 6))

for i, model_type in enumerate(model_types):
    steps = results_info[run][model_type]["steps"]
    train_accuracy = results_info[run][model_type]["train_accuracy"]
    
    plt.plot(steps, train_accuracy, label=model_labels[model_type], 
            color=model_colors[i], linewidth=2)

plt.title(f"Training Accuracy Over Training - {dataset_name.replace('_', ' ').title()}")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"train_accuracy_{dataset_name}.png", dpi=300)
plt.close()

# Plot 3: Model Size Comparison
plt.figure(figsize=(10, 6))

x = np.arange(len(model_types))
width = 0.35

original_sizes = []
quantized_sizes = []

for model_type in model_types:
    original_sizes.append(results_info[run][model_type]["model_size_kb"])
    quantized_sizes.append(results_info[run][model_type]["quantized_size_kb"])

bars1 = plt.bar(x - width/2, original_sizes, width, label='Original', 
                color='skyblue', edgecolor='black')
bars2 = plt.bar(x + width/2, quantized_sizes, width, label='Quantized', 
                color='lightcoral', edgecolor='black')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f}', ha='center', va='bottom')

plt.title(f"Model Size Comparison - {dataset_name.replace('_', ' ').title()}")
plt.xlabel("Model")
plt.ylabel("Model Size (KB)")
plt.xticks(x, [model_labels[m] for m in model_types])
plt.legend()
plt.grid(True, axis='y', ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig(f"model_size_{dataset_name}.png", dpi=300)
plt.close()

print("Plots saved successfully!")
print("Generated plots:")
print(f"  - train_accuracy_{dataset_name}.png")
print(f"  - val_accuracy_{dataset_name}.png")
print(f"  - model_size_{dataset_name}.png")