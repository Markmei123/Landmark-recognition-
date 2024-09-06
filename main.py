import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from Visiontransformer import ViT
from torch import nn, optim


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        # csv_file: 包含图像ID和标签的CSV文件路径
        # root_dir: 存储所有图像的根目录
        # transform: 图像预处理方法

        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # filter
        self.labels_df = self.labels_df[self.labels_df.apply(self._check_image_exists, axis=1)].reset_index(drop=True)

    def _check_image_exists(self, row):
        #check whether image exists
        img_id = row[0]
        img_path = os.path.join(self.root_dir, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
        return os.path.exists(img_path)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx, 0]  # get ID
        label = self.labels_df.iloc[idx, 2]  # get label

        # build path
        img_path = os.path.join(self.root_dir, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

print("1111")
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(csv_file='./dataset/train.csv', root_dir='./dataset/train', transform=transforms)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

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

def train(model, train_loader, criterion, optimizer, epochs=100):
    for imgs, labels in train_loader:
        print(imgs.shape, labels.shape)
        break
    model.train()
    for epoch in range(epochs):
        print(f"epoch:{epoch} in {epochs}")
        total_loss = 0
        correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {correct / len(train_loader.dataset)}")

train(model, train_loader, criterion, optimizer)



