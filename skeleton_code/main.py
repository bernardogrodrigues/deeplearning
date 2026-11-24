import torch
from torch import nn
from torch.utils.data import DataLoader

from load_data import CLASSES, load_coco_dominant10
from data import channel_stats, image_loaders

data = load_coco_dominant10()

# Structure of each split
for split in data:
    print(f"\n{split.upper()}")

    print(f"Images shape: {data[split]['images'].shape}")
    print(f"Labels shape: {data[split]['labels'].shape}")

    label_counts = torch.bincount(data[split]["labels"])

    print("Class distribution:")
    for class_idx, class_name in enumerate(CLASSES):
        print(f"    {class_name}: {label_counts[class_idx].item()}")

    if split == "train":
        mean, std = channel_stats(data[split]["images"])
        print(f"\nNormalization statistics (training set):")
        print(f"Channel mean: {mean}")
        print(f"Channel std: {std}")