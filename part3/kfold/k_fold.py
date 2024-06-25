import os
import pandas as pd
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
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
        print(f"Initial dataset length: {len(self.data)}")  # Debug: print initial dataset length
        self.transform = transform

        # Filter out missing files and print invalid paths
        invalid_paths = self.data[~self.data['image_path'].apply(os.path.exists)]
        for invalid_path in invalid_paths['image_path']:
            print(f"Invalid path: {invalid_path}")

        self.data = self.data[self.data['image_path'].apply(os.path.exists)].reset_index(drop=True)  # Filter out missing files
        print(f"Filtered dataset length: {len(self.data)}")  # Debug: print filtered dataset length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_path']
        image = Image.open(img_name)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.data.iloc[idx]['label_num_class']

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = FacialDataset(csv_file='../csv/kfold.csv', transform=transform)

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
    labels = dataset.data['label_num_class'].values
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

def run_k_fold_cross_validation(dataset, k_folds=10):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    results = []

    if len(dataset) == 0:
        print("The dataset is empty after filtering for valid image paths.")
        return

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{k_folds}')

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Further split train_subset into train and validation subsets
        train_size = int(0.85 * len(train_subset))
        val_size = len(train_subset) - train_size
        train_data, val_data = random_split(train_subset, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        class_weights = compute_class_weights(dataset)
        weights = torch.tensor([class_weights[i] for i in range(4)], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)

        model = FacialExpressionCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Train the model
        model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, patience=3)

        # Evaluate the model
        model.eval()
        test_acc = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_acc += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = test_acc.double() / len(test_loader.dataset)
        precision_macro = precision_score(all_labels, all_preds, average='macro')
        recall_macro = recall_score(all_labels, all_preds, average='macro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        precision_micro = precision_score(all_labels, all_preds, average='micro')
        recall_micro = recall_score(all_labels, all_preds, average='micro')
        f1_micro = f1_score(all_labels, all_preds, average='micro')

        print(f'Test Accuracy for fold {fold+1}: {test_acc:.4f}')
        print(f'Macro Precision for fold {fold+1}: {precision_macro:.4f}')
        print(f'Macro Recall for fold {fold+1}: {recall_macro:.4f}')
        print(f'Macro F1 Score for fold {fold+1}: {f1_macro:.4f}')
        print(f'Micro Precision for fold {fold+1}: {precision_micro:.4f}')
        print(f'Micro Recall for fold {fold+1}: {recall_micro:.4f}')
        print(f'Micro F1 Score for fold {fold+1}: {f1_micro:.4f}')

        fold_metrics.append({
            'fold': fold+1,
            'accuracy': test_acc.item(),
            'macro_precision': precision_macro,
            'macro_recall': recall_macro,
            'macro_f1_score': f1_macro,
            'micro_precision': precision_micro,
            'micro_recall': recall_micro,
            'micro_f1_score': f1_micro
        })
        results.append(test_acc.item())

    # Calculate average metrics
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_macro_precision = np.mean([m['macro_precision'] for m in fold_metrics])
    avg_macro_recall = np.mean([m['macro_recall'] for m in fold_metrics])
    avg_macro_f1 = np.mean([m['macro_f1_score'] for m in fold_metrics])
    avg_micro_precision = np.mean([m['micro_precision'] for m in fold_metrics])
    avg_micro_recall = np.mean([m['micro_recall'] for m in fold_metrics])
    avg_micro_f1 = np.mean([m['micro_f1_score'] for m in fold_metrics])

    # Print overall results
    print(f'K-Fold Cross Validation results:')
    for metrics in fold_metrics:
        print(f"Fold {metrics['fold']}: Accuracy: {metrics['accuracy']:.4f}, "
              f"Macro Precision: {metrics['macro_precision']:.4f}, Macro Recall: {metrics['macro_recall']:.4f}, Macro F1 Score: {metrics['macro_f1_score']:.4f}, "
              f"Micro Precision: {metrics['micro_precision']:.4f}, Micro Recall: {metrics['micro_recall']:.4f}, Micro F1 Score: {metrics['micro_f1_score']:.4f}")

    print(f'Average Test Accuracy: {avg_accuracy:.4f}')
    print(f'Average Macro Precision: {avg_macro_precision:.4f}')
    print(f'Average Macro Recall: {avg_macro_recall:.4f}')
    print(f'Average Macro F1 Score: {avg_macro_f1:.4f}')
    print(f'Average Micro Precision: {avg_micro_precision:.4f}')
    print(f'Average Micro Recall: {avg_micro_recall:.4f}')
    print(f'Average Micro F1 Score: {avg_micro_f1:.4f}')

    # Document results in a table format for the report
    report = pd.DataFrame(fold_metrics)
    report.loc['Average'] = ['Average', avg_accuracy, avg_macro_precision, avg_macro_recall, avg_macro_f1,
                             avg_micro_precision, avg_micro_recall, avg_micro_f1]
    print("\nTable 1: K-Fold Cross-Validation Results")
    print(report)

# Run 10-fold cross-validation
run_k_fold_cross_validation(dataset, k_folds=10)