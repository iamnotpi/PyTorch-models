import matplotlib.pyplot as plt
import torch

def plot_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.show()

def to_one_hot(labels, num_classes):
    one_hot_encoded = torch.zeros((labels.shape[0], num_classes))
    for label in range(labels.shape[0]):
        one_hot_encoded[label, int(labels[label].item())-1] = 1 # .item() get the scalar value from a 0-d tensor
    return one_hot_encoded

def accuracy(y_preds, y_test):
    correct = torch.eq(y_preds, y_test).sum().item()
    return (correct / len(y_preds)) * 100
