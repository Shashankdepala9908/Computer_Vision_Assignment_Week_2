import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define dataset transformation (normalize and convert to tensors)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load full CIFAR-10 dataset
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Define labels subset (3 classes)
selected_classes = [0, 1, 2]  # Airplane, Automobile, Bird

# Extract indices of selected classes
train_indices = [i for i, label in enumerate(full_trainset.targets) if label in selected_classes]
test_indices = [i for i, label in enumerate(full_testset.targets) if label in selected_classes]

# Reduce dataset size by limiting samples per class
max_samples_per_class = 2000  # Adjust to fit under 70MB

# Function to limit class size
def limit_class_size(indices, targets, max_samples):
    class_counts = {cls: 0 for cls in selected_classes}
    limited_indices = []
    for idx in indices:
        label = targets[idx]
        if class_counts[label] < max_samples:
            limited_indices.append(idx)
            class_counts[label] += 1
    return limited_indices

train_indices = limit_class_size(train_indices, full_trainset.targets, max_samples_per_class)
test_indices = limit_class_size(test_indices, full_testset.targets, max_samples_per_class // 2)  # Fewer test samples

# Create reduced datasets
train_data = np.array([full_trainset.data[i] for i in train_indices])
train_labels = [full_trainset.targets[i] for i in train_indices]

test_data = np.array([full_testset.data[i] for i in test_indices])
test_labels = [full_testset.targets[i] for i in test_indices]

# Convert to PIL Images (torchvision format)
train_data = [Image.fromarray(img) for img in train_data]
test_data = [Image.fromarray(img) for img in test_data]

# Define custom dataset class
class CustomCIFAR10(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Create PyTorch datasets
trainset = CustomCIFAR10(train_data, train_labels, transform=transform)
testset = CustomCIFAR10(test_data, test_labels, transform=transform)

# Display a sample image
plt.imshow(train_data[0])
plt.title(f"Sample Image - Class {train_labels[0]}")
plt.axis("off")
plt.show()
