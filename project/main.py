import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_data import CLASSES, load_coco_dominant10
from data import channel_stats, image_loaders
from train import train, get_device, set_seed
from models import MLP
from utils import test_score_fn, plot_history

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

# print (data["train"]["images"])

# Normalization
train_loader, val_loader, test_loader = image_loaders(data, batch_size=64, mean=mean, std=std)
# ---------------------------------------------- 2.1 ----------------------------------------------
# # Train the model
# set_seed(0)
# mlp_0d = MLP(dim_layers=[64*64*3, 10]) #where 0d stands for 0 hidden layers

# optimizer = torch.optim.SGD(mlp_0d.parameters(), lr=0.01)

# Check what device is being used (CPU or GPU)
print(get_device())

# criterion = nn.NLLLoss()

def loss_fn (model, batch): # batch is a tuple of (images, labels)
    predictions = model(batch[0])
    loss = criterion(predictions, batch[1])
    return loss

def val_score_fn(model):
    correct = 0
    total = 0

    for images, labels in val_loader:
        predictions = model(images)
        predicted_labels = torch.argmax(predictions, dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total

    return accuracy

# # Train the model
# history = train(mlp_0d, optimizer, loss_fn, train_loader, val_score_fn, epochs=20)

# # Plot the training loss and validation score
# plot_history(history)

# # Test accuracy of the best epoch model
# test_accuracy = test_score_fn(mlp_0d, test_loader)
# print (f"Test accuracy: {test_accuracy:.4f}")

#---------------------------------------------- 2.2 ----------------------------------------------


mlp_0d_2 = MLP(dim_layers=[64*64*3, 10]) #where 0d stands for 0 hidden layers

optimizers = [torch.optim.SGD(mlp_0d_2.parameters(), lr=0.01), 
                torch.optim.Adam(mlp_0d_2.parameters(), lr=0.01),
                  torch.optim.AdamW(mlp_0d_2.parameters(), lr=0.01)] 

results = []
mean, std = [], []

for opt in optimizers:
    schedulers = [None, torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1),
                    torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=20),
                        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20),
                            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20)]

    for sched in schedulers:
        val_score = []
        for seed in [0,1,2]:
            set_seed(seed)

            optimizer = opt
            scheduler = sched

            criterion = nn.NLLLoss()

            # Train the model
            history = train(mlp_0d_2, optimizer, loss_fn, train_loader, val_score_fn, epochs=20, scheduler=scheduler, patience=3)

            # results.append(history["val_score"][-1]) # append the last validation score of the training history
            val_score.append(history["val_score"][history["best_epoch"]])

        mean.append(torch.tensor(val_score, dtype=torch.float32).mean())
        std.append(torch.tensor(val_score, dtype=torch.float32).std())

print (f"Mean: {mean}")
print (f"Std: {std}")

# Plot the training loss and validation score
#plot_history(history)

# Test accuracy of the best epoch model
test_accuracy = test_score_fn(mlp_0d_2, test_loader)
print (f"Test accuracy: {test_accuracy:.4f}")