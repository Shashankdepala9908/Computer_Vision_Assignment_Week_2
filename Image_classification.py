import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define dataset transformation (normalize and convert to tensors)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define labels subset (3 classes)
selected_classes = [0, 1, 2]  # Example: Airplane, Automobile, Bird

# Filter dataset
trainset.data = trainset.data[[i for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]]
trainset.targets = [trainset.targets[i] for i in range(len(trainset.targets)) if trainset.targets[i] in selected_classes]

testset.data = testset.data[[i for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]]
testset.targets = [testset.targets[i] for i in range(len(testset.targets)) if testset.targets[i] in selected_classes]

# Display a sample image
plt.imshow(trainset.data[0])
plt.title(f"Sample Image - Class {trainset.targets[0]}")
plt.axis("off")
plt.show()