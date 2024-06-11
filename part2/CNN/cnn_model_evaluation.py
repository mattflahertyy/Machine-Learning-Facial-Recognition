import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]['label_num']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load test data from CSV
test_data = pd.read_csv('part2/csv_split_data/csv_fixed_label/test_data.csv')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create DataLoader
test_dataset = CustomDataset(test_data, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model class for best variant
class BestVariantCNN(nn.Module):
    def __init__(self):
        super(BestVariantCNN, self).__init__()
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
            nn.Linear(128, 4)  # 4 class emotions
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def calculate_fc_input_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            x = self.conv_layer(dummy_input)
            return x.numel()

# Define the model class for variant 1
class Variant1CNN(nn.Module):
    def __init__(self):
        super(Variant1CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
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
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            x = self.conv_layer(dummy_input)
            return x.numel()

# Define the model class for variant 2
class Variant2CNN(nn.Module):
    def __init__(self):
        super(Variant2CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=1),
            nn.BatchNorm2d(128),
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
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            x = self.conv_layer(dummy_input)
            return x.numel()

# Function to load model
def load_model(filepath, model_class):
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model

# Load models
main_model = load_model('part2/CNN/training_results/6_best_model_final.pth', BestVariantCNN)
variant1_model = load_model('part2/CNN/training_results/4_best_model_variant1.pth', Variant1CNN)
variant2_model = load_model('part2/CNN/training_results/5_best_model_variant2.pth', Variant2CNN)

# Function to evaluate model
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds

# Evaluate all models
labels, main_preds = evaluate_model(main_model, test_loader)
_, variant1_preds = evaluate_model(variant1_model, test_loader)
_, variant2_preds = evaluate_model(variant2_model, test_loader)

# Function to calculate metrics
def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    
    return accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1

# Calculate metrics for each model
main_metrics = calculate_metrics(labels, main_preds)
variant1_metrics = calculate_metrics(labels, variant1_preds)
variant2_metrics = calculate_metrics(labels, variant2_preds)

# Generate confusion matrices
def plot_confusion_matrix(labels, preds, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

plot_confusion_matrix(labels, main_preds, 'Main Model Confusion Matrix')
plot_confusion_matrix(labels, variant1_preds, 'Variant 1 Confusion Matrix')
plot_confusion_matrix(labels, variant2_preds, 'Variant 2 Confusion Matrix')

# Summarize findings in a table
metrics_table = pd.DataFrame({
    'Model': ['Main Model', 'Variant 1', 'Variant 2'],
    'Accuracy': [main_metrics[0], variant1_metrics[0], variant2_metrics[0]],
    'Macro Precision': [main_metrics[1], variant1_metrics[1], variant2_metrics[1]],
    'Macro Recall': [main_metrics[2], variant1_metrics[2], variant2_metrics[2]],
    'Macro F1': [main_metrics[3], variant1_metrics[3], variant2_metrics[3]],
    'Micro Precision': [main_metrics[4], variant1_metrics[4], variant2_metrics[4]],
    'Micro Recall': [main_metrics[5], variant1_metrics[5], variant2_metrics[5]],
    'Micro F1': [main_metrics[6], variant2_metrics[6], variant2_metrics[6]],
})

print(metrics_table)