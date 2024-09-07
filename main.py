import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torch.utils.data import DataLoader
from Visiontransformer import ViT
from torch import nn, optim
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.labels_df = self.labels_df[self.labels_df.apply(self._check_image_exists, axis=1)].reset_index(drop=True)

    def _check_image_exists(self, row):
        img_id = row[0]
        img_path = os.path.join(self.root_dir, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
        return os.path.exists(img_path)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 2]
        img_path = os.path.join(self.root_dir, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = CustomImageDataset(csv_file='./dataset/train.csv', root_dir='./dataset/train', transform=transform)

# Split into training and validation sets (80/20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=203093,
    dim=128,
    depth=6,
    heads=8,
    mlp_dim=256,
    dropout=0.1,
    emb_dropout=0.1
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    # Lists to store losses and accuracies
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = []

    # Variable to store the highest validation accuracy
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        train_correct = 0
        total_train_samples = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / total_val_samples
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Check if this is the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Output overall accuracy at the end of training
    print("Training completed.")
    print(f"Highest Validation Accuracy: {best_val_accuracy:.4f}")

    # Plot results
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.show()

# Train the model
train(model, train_loader, val_loader, criterion, optimizer, epochs=20)

