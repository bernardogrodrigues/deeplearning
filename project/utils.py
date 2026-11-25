import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test_score_fn(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images)
            predicted_labels = torch.argmax(predictions, dim=1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Plot the training loss and validation score
def plot_history(history):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(history['train_loss'], 'b-', label='Train loss')
    ax2.plot(history['val_score'], 'r-', label='Val score')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax2.set_ylabel('Validation score')
    ax2.set_ylim(0, 1)
    fig.legend(loc='center right', bbox_to_anchor=(0.9, 0.5))
    plt.show()