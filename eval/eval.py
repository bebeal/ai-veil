import os
import time
import json
import subprocess
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import timm
import nvidia_smi
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

IMAGENET_DIR = "/home/bebeal-desktop-ubuntu24/Datasets/ImageNet2012"
SPLITS = ["val"]
LABELS = json.load(open("/home/bebeal-desktop-ubuntu24/Datasets/ImageNet2012/ImageNet_class_index.json"))
LOG_DIR = f"logs/eval_base_imagenet2012" + "_" + SPLITS[0]
MODELS = [
    {"name": "convnextv2_tiny.fcmae_ft_in22k_in1k_384"},
    {"name": "mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k"},
    {"name": "vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k"},
    {"name": "resnet50d.ra4_e3600_r224_in1k"},
    {"name": "eva02_large_patch14_448.mim_in22k_ft_in1k"},
]

# Function to map label indices to human-readable class names
def get_class_name(index):
    return LABELS[str(index)][1] if index < len(LABELS) else "Unknown"

# Function to add text annotations on top of images using matplotlib
def annotate_images_with_labels(images, preds, labels):
    # Convert images from Tensor to numpy array for processing with matplotlib
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, C) format
    
    fig, ax = plt.subplots(1, len(images_np), figsize=(15, 5))
    if len(images_np) == 1:
        ax = [ax]
        
    for i, image in enumerate(images_np):
        image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1]
        ax[i].imshow(image)
        ax[i].axis('off')
        ax[i].set_title(f"Pred: {get_class_name(preds[i].item())}\nActual: {get_class_name(labels[i].item())}", fontsize=12)
    
    plt.tight_layout()
    fig.canvas.draw()
    # annotated_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # matplotlib says use buffer_rgba instead of tostring_rgb, but its not 1:1, buffer_rgba adds a 4th alpha channel unexpectedly, so account for that
    annotated_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    annotated_image = annotated_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    annotated_image = annotated_image[:, :, :3]  # Remove the alpha channel to get RGB
    plt.close(fig)
    return annotated_image

# Get device to run model on
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Setup Nvidia SMI for VRAM statistics
def setup_nvidia_smi():
    nvidia_smi.nvmlInit()
    return nvidia_smi.nvmlDeviceGetHandleByIndex(0)

# Log GPU VRAM usage in a single graph with different colors for each model
def log_vram_usage(gpu_handle, writer, model_name, split, epoch=0):
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
    total_vram = info.total // (1024 ** 2)  # Convert to MB
    used_vram = info.used // (1024 ** 2)    # Convert to MB
    vram_percentage = used_vram / total_vram
    writer.add_scalars(f"VRAM/MB/{split}", {model_name: used_vram}, global_step=epoch)
    writer.add_scalars(f"VRAM/%/{split}", {model_name: round(vram_percentage * 100, 2)}, global_step=epoch)
    return used_vram

# Function to create dataloaders for the given dataset directory
# and model to fetch the data config and transforms
def create_dataloader(split, model, batch_size=1):
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    dataset = datasets.ImageNet(root=IMAGENET_DIR, split=split, transform=transforms)
    # pin_memory=True for faster data transfer to GPU: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
    # getting connection issues with persistent_workers=True, so disabling it for now
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

def create_model(config, model_name):
    try:
      model = timm.create_model(model_name, pretrained=True).to(config["device"])
      model.eval()
      return model
    except Exception as e:
      print(f"Error creating model {model_name}: {e}")
      return None

# Start TensorBoard automatically within the script
def start_tensorboard(log_dir, port=6006):
    subprocess.run(f"fuser -k {port}/tcp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Starting TensorBoard at http://localhost:{port}/")
    subprocess.Popen(f"tensorboard --logdir={log_dir} --host=0.0.0.0 --port={port} --load_fast=false", shell=True)
    time.sleep(1)

def evaluate_model(config, model_name):
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(LOG_DIR + '/profiler'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as profiler:
    writer = config["writer"]
    print('-'*50, '\nEvaluating model:', model_name)
    model = create_model(config, model_name)
    with torch.no_grad():
        for split in SPLITS:
            total_correct = 0
            total_images = 0
            log_vram_usage(config["gpu_handle"], config["writer"], model_name, split, 0)
            dataloader = create_dataloader(split, model, batch_size=config["batch_size"])
            if dataloader is None:
                print(f"Skipping split {split} as it is not available in the dataset")
                continue
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches", leave=True, position=0, ncols=50):
                inputs, labels = batch
                inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_images += labels.size(0)
                if batch_idx == 0:
                    log_vram_usage(config["gpu_handle"], config["writer"], model_name, split, 1)
            # profiler.step()
            accuracy = (total_correct / total_images) * 100
            print(f"  Accuracy on {split}: {accuracy} %")
            writer.add_scalars(f"accuracy/{split}", {model_name: accuracy}, global_step=0)
        print('-'*50, '\n')
        del model
        torch.cuda.empty_cache()

def evaluate_models(config):
    for model in MODELS:
        evaluate_model(config, model["name"])

def main():
    writer = SummaryWriter(log_dir=LOG_DIR)
    start_tensorboard(LOG_DIR)
    gpu_handle = setup_nvidia_smi()
    device = get_device()
    config = {
        "gpu_handle": gpu_handle,
        "device": device,
        "writer": writer,
        "batch_size": 150,
    }

    # Run evaluation on models
    evaluate_models(config)
    
    writer.close()
    nvidia_smi.nvmlShutdown()
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
