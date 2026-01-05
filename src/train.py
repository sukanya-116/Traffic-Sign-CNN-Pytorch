# =========================
# Imports & Setup
# =========================

import os
import glob
import kagglehub
import torch
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


# =========================
# Dataset Analysis
# =========================

def analyze_dataset(root_dir):
    dataset_info = {}
    class_counts = defaultdict(int)  # track image count per class folder

    for root, _, files in os.walk(root_dir):
        rel_dir = os.path.relpath(root, root_dir)

        if rel_dir == ".":  # skip root folder itself
            continue

        file_counts = defaultdict(int)

        for file in files:
            ext = os.path.splitext(file)[1].lower()

            if ext in ['.jpg', '.jpeg', '.png']:  # count only images
                file_counts[ext] += 1
                class_counts[rel_dir] += 1

        if file_counts:
            dataset_info[rel_dir] = dict(file_counts)

    return dataset_info, class_counts



# =========================
# Classes
# =========================

classes = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Veh > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing veh > 3.5 tons'
}

num_classes=43


# =========================
# Custom Dataset
# =========================

class GTSRBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.classes = sorted(os.listdir(data_dir), key=lambda x: int(x))
        self.class_to_idx = {cls: int(cls) for cls in self.classes}

        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# Transforms
# =========================

def get_transforms():
   
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform

# =========================
# Data Loaders
# =========================

def get_data_loaders(train_dir, batch_size=32):
    train_transform, test_transform = get_transforms()
    full_train_dataset = GTSRBDataset(
        train_dir,
        transform=None
    )
    train_size = int(0.85 * len(full_train_dataset))
    val_size   = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # Override transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform   = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# =========================
# Model
# =========================

class GTSRB(nn.Module):
    def __init__(self, size_inner=20, droprate=0.2, num_classes=num_classes):
        super().__init__()

        # Load pre-trained Efficientnet_B0
        self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Remove original classifier
        self.base_model.classifier = nn.Identity()

        # Add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)  # Add dropout
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.output_layer(x)
        return x

def make_model(learning_rate=0.001, size_inner=20,droprate=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GTSRB(size_inner=size_inner,num_classes=43,droprate=droprate)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

# =========================
# Training
# =========================

def train_and_evaluate_with_checkpoint(model, optimizer, train_loader, val_loader, criterion, num_epochs, device):
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint_path = f'efficientnet_b0_{epoch+1:02d}_{val_acc:.3f}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')

    return model, history


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================
    # Dataset Download
    # =========================
    
    print("Downloading dataset...")
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    print("Path to dataset files:", path)

    TRAIN_DIR = os.path.join(path, "Train")
    
    learning_rate = 0.001
    drop_rate = 0.2
    inner_layer_size = 20
    num_epochs = 20
    
    dataset_structure, class_counts = analyze_dataset(path)

    # Print folder → file type count
    print("Dataset Structure:")
    for folder, types in dataset_structure.items():
        print(f"{folder} → {types}")

    # =========================
    # DataLoaders
    # =========================

    train_loader, val_loader = get_data_loaders(TRAIN_DIR, batch_size=32)

    
    model, optimizer = make_model(
        learning_rate=learning_rate, 
        droprate=drop_rate, 
        size_inner=inner_layer_size
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_and_evaluate_with_checkpoint(
        model, 
        optimizer, 
        train_loader, 
        val_loader, 
        criterion, 
        num_epochs, 
        device
    )

if __name__ == "__main__":
    main()