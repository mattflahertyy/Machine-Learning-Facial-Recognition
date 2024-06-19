import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


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


# Load test data
test_data = pd.read_csv('..//csv_split_data/csv_fixed_label/test_data_final.csv')

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# DataLoader
test_dataset = CustomDataset(test_data, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Function to load model and evaluate
def load_and_evaluate_model(model_filepath, test_loader):
    model = torch.load(model_filepath)  # Load model directly from .pth file
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


# List of model filepaths
model_files = {
    'Main Model': './training_results_final/main_model_final.pth',
    'Variant 1': './training_results_final/variant_1_final.pth',
    'Variant 2': './training_results_final/variant_2_final.pth'
}

# Evaluate all models
metrics = {}

for model_name, model_filepath in model_files.items():
    labels, preds = load_and_evaluate_model(model_filepath, test_loader)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')

    metrics[model_name] = {
        'Accuracy': accuracy,
        'Macro Precision': precision,
        'Macro Recall': recall,
        'Macro F1': f1,
        'Micro Precision': micro_precision,
        'Micro Recall': micro_recall,
        'Micro F1': micro_f1
    }

# Display metrics
metrics_table = pd.DataFrame(metrics).transpose()
print(metrics_table)


# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# Plot confusion matrices
for model_name, model_filepath in model_files.items():
    labels, preds = load_and_evaluate_model(model_filepath, test_loader)
    plot_confusion_matrix(labels, preds, f'{model_name} Confusion Matrix')
