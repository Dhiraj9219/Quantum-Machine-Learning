# ============================================
# 0. Extract ZIP (if not already extracted)
# ============================================
import zipfile, os

zip_path = "/content/drive/MyDrive/archive.zip"
extract_path = "/content"

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_path)
print("Dataset extracted!")

# Dataset path is direct: /content/train and /content/test
DATASET_ROOT = "/content"
print("Using dataset folder:", DATASET_ROOT)


# ============================================
# Full Hybrid CNN -> QCNN (PennyLane + PyTorch)
# ============================================
import os, time, random
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pennylane as qml
from pennylane import numpy as pnp
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# USER PARAMETERS
# ---------------------------
IMG_SIZE = 64
MAX_PER_CLASS_TRAIN = 2000
MAX_PER_CLASS_TEST  = 500
BATCH_SIZE = 8
EPOCHS = 40
LR = 3e-4
SEED = 42
n_qubits = 8
n_q_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------------------------------------
# 1. DATASET CLASS — WITH MERGED LABELS (FINAL VERSION)
# -----------------------------------------------------
class MRIDataset(Dataset):
    def __init__(self, root, split="train", img_size=64, max_per_class=None, augment=False):
        self.imgs = []
        self.labels = []

        split_root = os.path.join(root, split)
        classes = sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))])

        for label, cls in enumerate(classes):
            cls_folder = os.path.join(split_root, cls)
            files = sorted([os.path.join(cls_folder, f)
                            for f in os.listdir(cls_folder)
                            if f.lower().endswith((".jpg",".jpeg",".png"))])

            files = files[:max_per_class] if max_per_class else files

            for f in files:
                self.imgs.append(f)

                # MERGED LABELS:
                # 0,1 → 0 (Non-Demented + Very Mild)
                # 2,3 → 1 (Mild + Moderate)
                if label in [0, 1]:
                    new_label = 0
                else:
                    new_label = 1

                self.labels.append(new_label)

        self.img_size = img_size
        self.augment = augment

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self.aug_trans = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("L")
        if self.augment:
            img = self.aug_trans(img)

        img = self.transform(img)
        return img, label


# ---------------------------
# 2. LOADERS
# ---------------------------
train_ds = MRIDataset(DATASET_ROOT, split="train", augment=True, max_per_class=MAX_PER_CLASS_TRAIN)
test_ds  = MRIDataset(DATASET_ROOT, split="test", augment=False, max_per_class=MAX_PER_CLASS_TEST)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train samples:", len(train_ds))
print("Test samples:", len(test_ds))


# ---------------------------
# 3. SMALL CNN FEATURE EXTRACTOR
# ---------------------------
class SmallCNN(nn.Module):
    def __init__(self, out_dim=n_qubits):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.fc(x)


# ---------------------------
# 4. QUANTUM LAYER
# ---------------------------
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_q_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)


# ---------------------------
# 5. HYBRID MODEL
# ---------------------------
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SmallCNN()
        self.qlayer = qlayer
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, x):
        f = self.cnn(x)
        scaled = torch.tanh(f) * (np.pi/2 * 0.95)
        q_out = self.qlayer(scaled)
        return self.classifier(q_out)


# ---------------------------
# 6. TRAINING SETUP
# ---------------------------
model = HybridModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("Total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------
# 7. TRAINING LOOP
# ---------------------------
def evaluate(model, loader):
    model.eval()
    preds, targs = [], []
    loss_total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_total += criterion(out, yb).item() * xb.size(0)
            preds.append(torch.argmax(out, dim=1).cpu().numpy())
            targs.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    return loss_total/len(loader.dataset), accuracy_score(targs, preds), preds, targs


best_acc = 0
best_state = None

print("Training QCNN...")

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    val_loss, val_acc, _, _ = evaluate(model, test_loader)
    print(f"Epoch {epoch}/{EPOCHS}  |  Train Loss: {train_loss/len(train_ds):.4f}  |  Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = model.state_dict()


model.load_state_dict(best_state)


# ---------------------------
# 8. FINAL EVALUATION
# ---------------------------
loss, acc, preds, targs = evaluate(model, test_loader)

print("\nFINAL TEST ACCURACY:", acc)
print("\nCLASSIFICATION REPORT:\n",
      classification_report(targs, preds, target_names=["Healthy/Very-Mild", "Mild/Moderate"]))
print("\nCONFUSION MATRIX:\n", confusion_matrix(targs, preds))
