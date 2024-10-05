import time
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import collections
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib import font_manager

def parse_tf_events(events):
    size_guidance = {
        event_accumulator.SCALARS: 0,
    }
    ea = event_accumulator.EventAccumulator(events, size_guidance)
    ea.Reload()

    tags = ea.Tags()
    scalar_tags = tags['scalars']
    data = collections.defaultdict(list)
    
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        data[tag] = [(scalar.step, scalar.value) for scalar in scalar_events]

    return data

def extract_model_name(file_path):
    for model in models_to_evaluate:
        if model in file_path:
            return model
    return None

models_to_evaluate = [
    "resnet50d.ra4_e3600_r224_in1k",
    "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k",
    "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    "vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k",
    "eva02_large_patch14_448.mim_in22k_ft_in1k"
]

tags = ["accuracy/val", "accuracy/train"]
data_buckets = {model: {tag: [] for tag in tags} for model in models_to_evaluate}

root_log_dirs = ["./eval/archive/eval_base_imagenet2012_val", "./eval/archive/eval_base_imagenet2012_train"]

tfevents = []
for root_log_dir in root_log_dirs:
    if not os.path.exists(root_log_dir):
        continue
    log_dirs = [os.path.join(root_log_dir, directory) for directory in os.listdir(root_log_dir) if os.path.isdir(os.path.join(root_log_dir, directory))]
    for directory in log_dirs:
        tfevents += [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith("events.out")]

for events in tfevents:
    model_name = extract_model_name(events)
    if model_name is None:
        continue
    parsed_data = parse_tf_events(events)
    for tag, data in parsed_data.items():
        if tag in data_buckets[model_name]:
            data_buckets[model_name][tag] += data

sorted_models = sorted(models_to_evaluate, 
                       key=lambda m: max(dict(data_buckets[m]["accuracy/val"]).values()), 
                       reverse=True)

# Font setup
home_dir = os.path.expanduser("~")
font_dir = os.path.join(home_dir, "assets", "fonts", "BerkeleyMono")
font_path = os.path.join(font_dir, "BerkeleyMono-Bold.ttf")  # Change to bold variant
# Print what's in the directory
if os.path.exists(font_path):
    font_properties = font_manager.FontProperties(fname=font_path)
    print(f"Using custom font: {font_properties.get_name()} (Bold)")
else:
    print(f"Bold font not found: {font_path}")
    print("Using default font")
    font_properties = None

# Plot setup
plt.style.use('dark_background')
sns.set_style("darkgrid", {'axes.facecolor': '.15', 'grid.color': '.8'})

gradient_colors = plt.cm.plasma
fig, ax = plt.subplots(figsize=(14, 10), dpi=900, facecolor='#1E1E1E')
ax.set_facecolor('#1E1E1E')

# Set up x-ticks for train and val categories
x_positions = np.array([0, 1.5])
x_labels = ['train', 'val']

# Parameters to control the width and spacing of the columns
bar_width = 0.10
space_between = 0.02

# Plot each model's data for train and val
for model_idx, model in enumerate(sorted_models):
    y_train = max(dict(data_buckets[model]["accuracy/train"]).values())
    y_val = max(dict(data_buckets[model]["accuracy/val"]).values())
    
    gradient = gradient_colors(np.linspace(0, 1, len(sorted_models)))[model_idx]
    
    for x, y in zip(x_positions, [y_train, y_val]):
        # Position each bar correctly
        bar_x = x + model_idx * (bar_width + space_between)  # Proper x-axis position
        bar = ax.bar(bar_x, y, width=bar_width, color=gradient, 
                     edgecolor='white', linewidth=1)
        
        # Place the text directly on top of the bar using ha='center'
        text = ax.text(bar_x, y + 0.5, f'{y:.2f}%', 
                       ha='center', va='bottom', fontweight='bold', fontsize=12, color='white', fontproperties=font_properties)
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])


# legend and cutoff dataset names
simplified_names = [model.split(".")[0].replace("_", " ") for model in sorted_models]
legend_elements = [plt.Rectangle((0,0),1,1, fc=gradient_colors(i/(len(sorted_models)-1))) for i in range(len(sorted_models))]
legend = ax.legend(legend_elements, simplified_names, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.20), ncol=2,
                   frameon=False, labelcolor='white', prop=font_properties,
                   handlelength=2, handleheight=2)
plt.setp(legend.get_texts(), fontsize=18)

# Stats
gpu_info = {
    "GPU": "NVIDIA 3080 Ti",
    "VRAM": "12288",
    "NVIDIA-SMI": "550.107.02",
    "CUDA": "12.4",
    "timm": "1.0.9",
    "torch": "2.4.1",
    "torchvision": "0.19.1",
}

# Format the stats with right-aligned values
formatted_text = "\n".join([
    f"{'GPU:':<15}{gpu_info['GPU']:>20}",
    f"{'VRAM:':<15}{gpu_info['VRAM']:>20}",
    f"{'NVIDIA-SMI:':<15}{gpu_info['NVIDIA-SMI']:>20}",
    f"{'CUDA:':<15}{gpu_info['CUDA']:>20}",
    f"{'timm:':<15}{gpu_info['timm']:>20}",
    f"{'torch:':<15}{gpu_info['torch']:>20}",
    f"{'torchvision:':<15}{gpu_info['torchvision']:>20}"
])

# Add text box of stats
ax.text(0.5, 0.5, formatted_text, fontsize=14, fontweight='bold', color='white',
        ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white'),
        transform=ax.transAxes, fontproperties=font_properties)

# title and labels
fig.suptitle('Baseline Model Evaluation\nImageNet 2012 Accuracy (train/val)', fontsize=20, fontweight='bold', color='white', y=1.0)
ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold', color='white', fontproperties=font_properties)
ax.set_ylim(80, 100)
ax.set_yticks(np.arange(80, 101, 5))
ax.set_yticklabels([f'{y}%' for y in range(80, 101, 5)], fontsize=16, color='white', fontproperties=font_properties)
ax.xaxis.grid(False)
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_color('none')
for x, label in zip(x_positions, x_labels):
    center = x + (len(sorted_models) - 1) * (bar_width + space_between) / 2
    ax.text(center, ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            label, ha='center', va='top', fontsize=16, color='white', 
            fontproperties=font_properties, transform=ax.transData)
fig.subplots_adjust(bottom=0.2)

# Save and close the plot
plt.tight_layout()
plt.savefig("./plots/base_imagenet2012_comparison.png", dpi=900, bbox_inches='tight', facecolor='#1E1E1E')
plt.close()

print("Revised plot saved as: base_imagenet2012_comparison.png")
