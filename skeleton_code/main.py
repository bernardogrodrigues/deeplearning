import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from load_data import CLASSES, load_coco_dominant10
from data import ImageDataset, CaptionDataset, channel_stats, image_loaders

data = load_coco_dominant10()

# Check the data structure
data_type = type(data)
dict_keys = data.keys()
data_shape = data["train"]["images"].shape
type_data = data["train"]["images"].dtype
label_shape = data["train"]["labels"].shape
label_counts = torch.bincount(data["train"]["labels"])

print (f"Data type: {data_type}")
print (f"Dictionary keys: {dict_keys}")
print (f"Data shape: {data_shape}")
print (f"Data type: {type_data}")
print (f"Label shape: {label_shape}")
print (f"Label counts: {label_counts}")