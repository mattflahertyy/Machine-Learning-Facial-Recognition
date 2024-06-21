import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# ---------------------------ADDING THE SEED ---------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# ----------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        label_race = self.data.iloc[idx]['label_num_race']

        if self.transform:
            image = self.transform(image)

        return image, label_race


# load test data
test_data = pd.read_csv('../csv/test.csv')

# transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# DataLoader
test_dataset = CustomDataset(test_data, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# main model
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
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
            nn.Linear(128, 4)  # Adjusted to 4 for previous model output size
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


# function to load model
def load_model(filepath, model_class):
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    model.eval()  #setting the model to evaluation mode
    return model


# load model
main_model = load_model('../models_P2/main_model.pth', MainModel)


# function to evaluate model
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels_race in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_race.cpu().numpy())

    return all_labels, all_preds


# evaluate model
labels, main_preds = evaluate_model(main_model, test_loader)


# function to calculate metrics
def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    return accuracy, precision, recall, f1


# calculate metrics for each race category
def race_metrics(labels, preds):
    race_dict = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Hispanic'}
    metrics = {}

    for race in np.unique(labels):
        race_labels = [label for label in labels if label == race]
        race_preds = [pred for label, pred in zip(labels, preds) if label == race]
        accuracy, precision, recall, f1 = calculate_metrics(race_labels, race_preds)
        metrics[race_dict[race]] = {
            'Num Images': len(race_labels),
            'Accuracy': accuracy*100,
            'Precision': precision*100,
            'Recall': recall*100,
            'F1 Score': f1*100
        }

    return metrics


# calculate race metrics
race_metrics = race_metrics(labels, main_preds)

# display metrics
for race, metrics in race_metrics.items():
    print(f"Metrics for {race}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
