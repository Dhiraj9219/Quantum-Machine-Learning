!pip install torch torchvision --quiet

# ===============================================================
# 1. MOUNT GOOGLE DRIVE
# ===============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile

# ===============================================================
# 2. EXTRACT ZIP (archive.zip)
# ===============================================================
zip_path = "/content/drive/MyDrive/archive.zip"

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("/content")
print("Done!")

print("Root folders:", os.listdir("/content"))

# ===============================================================
# 3. IMPORTS
# ===============================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===============================================================
# 4. CUSTOM DATASET WITH CLASS MERGING
# ===============================================================
class DementiaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir

        # Folder names EXACT in your dataset
        self.class_map = {
            "Non Demented": 0,
            "Very mild Dementia": 0,
            "Mild Dementia": 1,
            "Moderate Dementia": 1
        }

        self.paths = []
        self.labels = []

        for folder in os.listdir(root_dir):
            full = os.path.join(root_dir, folder)
            if not os.path.isdir(full):
                continue

            label = self.class_map.get(folder)
            if label is None:
                print("Skipping:", folder)
                continue

            for img in os.listdir(full):
                if img.lower().endswith(("jpg", "jpeg", "png")):
                    self.paths.append(os.path.join(full, img))
                    self.labels.append(label)

        print(f"Loaded {len(self.paths)} images from {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ===============================================================
# 5. DATA TRANSFORMS (AUGMENTATION)
# ===============================================================
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(128, scale=(0.8,1.0)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

train_path = "/content/train"
test_path  = "/content/test"

train_ds = DementiaDataset(train_path, train_transform)
test_ds  = DementiaDataset(test_path,  test_transform)


# ===============================================================
# 6. CREATE BALANCED SAMPLER
# ===============================================================
class_counts = np.bincount(train_ds.labels)
class_weights = 1.0 / class_counts

sample_weights = [class_weights[label] for label in train_ds.labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

# ===============================================================
# 7. RESNET18 MODEL
# ===============================================================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)  # 2 classes
model = model.to(device)

# Weighted Loss
weights = torch.tensor([1/class_counts[0], 1/class_counts[1]], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ===============================================================
# 8. TRAINING LOOP
# ===============================================================
EPOCHS = 10
print("\nTraining...\n")

for epoch in range(EPOCHS):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} "
          f"| Train Acc: {100*correct/total:.2f}%")

# ===============================================================
# 9. TESTING
# ===============================================================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs.to(device))
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nTEST ACCURACY:", 100 * np.mean(np.array(y_true)==np.array(y_pred)), "%")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["NonDemented", "Demented"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
