# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eUXfoLiRKpSCdzMkaiY7_0BoCXjjfJDH
"""

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


class FacialDataset(Dataset):
    # Here we are loading out CSV file containing image paths and labels
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    #return length of dataset
    def __len__(self):
        return len(self.data)

    # Getting image path and label for the given index
    def __getitem__(self, idx):
        img_name = os.path.join('../', self.data.iloc[idx, 1]) # going back 1 space
        image = Image.open(img_name)

        # Ensures all images are converted to RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.data.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)), #resizing images to the proper size
    # transforms.RandomHorizontalFlip(), #we can add these if we want to prevent overfititng
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = FacialDataset(csv_file='../csv_split_data/csv_fixed_label/train_data.csv', transform=transform)
val_dataset = FacialDataset(csv_file='../csv_split_data/csv_fixed_label/validation_data.csv', transform=transform)
test_dataset = FacialDataset(csv_file='../csv_split_data/csv_fixed_label/test_data.csv', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



class FacialExpressionCNN(nn.Module):
    def __init__(self):
        super(FacialExpressionCNN, self).__init__()

        #create sequentially ordered layers in the network for more modularity
        # CNN with 3 convolutional layers, grouped into 2 sets with batch normalization and leakyReLU activation
        #max pooling is applied after all convolutional layers
        # three fully connected layers and a dropout layer to prevent overfitting
        # outputs 4 classes

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, padding=1),  # 1st conv layer with 2x2 kernel
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1),  # 2nd conv layer with 2x2 kernel
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=1),  # 3rd conv layer with 2x2 kernel
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_input_size = self.calculate_fc_input_size()

        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 4)  # 4 class emotions
      )


    def forward(self, x):
        #convolutional layers
        x = self.conv_layer(x)
        # print(f'After conv_layer: {x.size()}')  # Print the size after convolutional layers -----------------------------


        #flattening
        x = x.view(x.size(0),-1)
        # print(f'After flattening: {x.size()}')  # Print the size after convolutional layers -----------------------------

        #fully connected layers
        x = self.fc_layer(x)
        # print(f'After fc_layer: {x.size()}')  # Print the size after convolutional layers -----------------------------
        return x

    def calculate_fc_input_size(self):
        #function to calculate the input size to the fully connected layers
        dummy_input = torch.randn(1, 3, 224, 224)  # Assuming input image size is 224x224
        x = self.conv_layer(dummy_input)
        fc_input_size = x.size(1) * x.size(2) * x.size(3)
        return fc_input_size

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, patience=3):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

     #train loop over the training dataset
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # print(f'Input size: {inputs.size()}')  # Print the input size  -----------------------------
            outputs = model(inputs) #forward pass
            # print(f'Output size: {outputs.size()}') -----------------------------
            loss = criterion(outputs, labels)
            loss.backward() #backward propagation
            optimizer.step() #perform the optimizer training step
            running_loss += loss.item() * inputs.size(0) #accumulate loss

        epoch_loss = running_loss / len(train_loader.dataset)

        #Monitor performance on the validation set
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

        #implementing early stopping technique based on the results from val loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0 #resetting when val loss improves
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1 #increasing when val loss doesnt improve

        if patience_counter >= patience:
            print("Early stopping")
            break #stop training if patience limit is hit

    model.load_state_dict(best_model_wts)
    return model

model = FacialExpressionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model = train_model(model, train_loader, val_loader, criterion, optimizer)


#save the best model
# torch.save(model.state_dict(), 'best_model.pth')