import os
import pandas as pd
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from collections import Counter

# ---------------------------ADDING THE SEED ---------------------------
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# ----------------------------------------------------------------------

class FacialDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 1]
        image = Image.open(img_name)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.data.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = FacialDataset(csv_file='../csv_split_data/csv_fixed_label/train_data_final.csv', transform=transform)
val_dataset = FacialDataset(csv_file='../csv_split_data/csv_fixed_label/validation_data_final.csv', transform=transform)
test_dataset = FacialDataset(csv_file='../csv_split_data/csv_fixed_label/test_data_final.csv', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class FacialExpressionCNN(nn.Module):
    def __init__(self):
        super(FacialExpressionCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_input_size = self.calculate_fc_input_size()

        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def calculate_fc_input_size(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        x = self.conv_layer(dummy_input)
        fc_input_size = x.size(1) * x.size(2) * x.size(3)
        return fc_input_size

def compute_class_weights(dataset):
    labels = dataset.data.iloc[:, 3].values
    class_counts = Counter(labels)
    total_count = len(labels)
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    return class_weights

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=3):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for inputs, labels in val_loader:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_acc += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_acc.double() / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model_wts)
    return model

class_weights = compute_class_weights(train_dataset)
weights = torch.tensor([class_weights[i] for i in range(4)], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)

model = FacialExpressionCNN()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, patience=3)

# Save the best model
# torch.save(model.state_dict(), 'best_model.pth')
