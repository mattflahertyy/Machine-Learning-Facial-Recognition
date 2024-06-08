import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from PIL import Image
import os

# Define the model
class MultiLayerFCNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultiLayerFCNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return F.log_softmax(x, dim=1)

# Load the CSV files
train_csv = pd.read_csv("csv_split_data/train_data.csv")
val_csv = pd.read_csv("csv_split_data/validation_data.csv")

# Define a custom dataset class
class CustomDataset(td.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = self.data['label_num'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 3] - 1  # Adjust labels to start from 0

        # Skip .DS_Store files
        if img_path.endswith('.DS_Store'):
            return None, None

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

# Define transformations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.225, 0.225, 0.225])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),
    normalize
])

# Create datasets
train_dataset = CustomDataset(csv_file="csv_split_data/train_data.csv", transform=transform)
val_dataset = CustomDataset(csv_file="csv_split_data/validation_data.csv", transform=transform)

# Remove None elements from the dataset
train_dataset.data = train_dataset.data[train_dataset.data['image_path'] != '.DS_Store']
val_dataset.data = val_dataset.data[val_dataset.data['image_path'] != '.DS_Store']

# Remove None elements from the dataset
train_dataset.data.dropna(inplace=True)
val_dataset.data.dropna(inplace=True)

# Create data loaders
batch_size = 64
train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # Add drop_last=True
val_loader = td.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # Add drop_last=True

# Calculate the input size dynamically
sample_input = train_dataset[0][0]
input_size = sample_input.shape[0] * sample_input.shape[1] * sample_input.shape[2]

# Define the hidden size
hidden_size = 100  # You can adjust this value based on experimentation

# Define the output size based on the number of unique labels
output_size = len(train_dataset.labels)

# Initialize the model
model = MultiLayerFCNet(input_size, hidden_size, output_size)

# Print the model summary
print(model)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        if images is None or labels is None:
            continue  # Skip NoneType elements
        images = images.view(images.size(0), -1)  # Flatten the images
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Validation loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        if images is None or labels is None:
            continue  # Skip NoneType elements
        images = images.view(images.size(0), -1)  # Flatten the images
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
