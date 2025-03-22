import numpy as np
import torch

import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image

# Define dataset transformation
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale to reduce dimensions
    transforms.Resize((16, 16)),  # Reduce image size
    transforms.ToTensor()
])

# Load CIFAR-10 dataset
data_path = "./data"
full_trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
full_testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)

# Select specific classes (Airplane, Automobile, Bird)
selected_classes = [0, 1, 2]
train_indices = [i for i, label in enumerate(full_trainset.targets) if label in selected_classes]
test_indices = [i for i, label in enumerate(full_testset.targets) if label in selected_classes]

# Limit dataset size
max_samples_per_class = 2000  # Reduce dataset size to keep under 70MB

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
test_indices = limit_class_size(test_indices, full_testset.targets, max_samples_per_class // 2)

# Create reduced datasets
train_data = [full_trainset.data[i] for i in train_indices]
train_labels = [full_trainset.targets[i] for i in train_indices]

test_data = [full_testset.data[i] for i in test_indices]
test_labels = [full_testset.targets[i] for i in test_indices]

# Convert to PIL Images for transformation
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

# Convert dataset to NumPy arrays for SVM training
X_train = np.array([img.numpy().flatten() for img, _ in trainset])
y_train = np.array(train_labels)

X_test = np.array([img.numpy().flatten() for img, _ in testset])
y_test = np.array(test_labels)

# Apply PCA to reduce features
pca = PCA(n_components=100)  # Reduce dimensions to 100
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVM classifier
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_pca, y_train)

# Predict and evaluate
y_pred_svm = svm.predict(X_test_pca)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print(f"SVM Accuracy: {svm_accuracy:.4f}")

from sklearn.linear_model import LogisticRegression

# Train Softmax classifier
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax.fit(X_train, y_train)

# Predict and evaluate
y_pred_softmax = softmax.predict(X_test)
softmax_accuracy = accuracy_score(y_test, y_pred_softmax)

print(f"Softmax Accuracy: {softmax_accuracy:.4f}")