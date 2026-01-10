# ================================================
# STEP 1: Import Libraries
# ================================================
import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os, cv2, random, time

# ================================================
# STEP 2: Mount Google Drive (if in Colab)
# ================================================
from google.colab import drive
drive.mount('/content/drive')

data_root = "/content/drive/MyDrive/archive"

# ================================================
# STEP 3: Load Images Safely
# ================================================
def load_images_from_folder(base_path, limit_per_class=75, img_size=64):
    X, y = [], []
    dementia_classes = ["Mild Dementia", "Moderate Dementia", "Very mild Dementia"]
    non_demented_class = "Non Demented"

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"❌ Dataset base path not found: {base_path}")

    for subset in ["train", "test"]:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path):
            continue

        print(f"\n📂 Loading {subset} data...")

        # Non-Demented → label 0
        nd_path = os.path.join(subset_path, non_demented_class)
        if os.path.exists(nd_path):
            nd_files = [f for f in os.listdir(nd_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(nd_files)
            for img_name in nd_files[:limit_per_class]:
                img_path = os.path.join(nd_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (img_size, img_size))
                X.append(img.flatten()); y.append(0)

        # Demented → label 1 (all types merged)
        for dementia_type in dementia_classes:
            d_path = os.path.join(subset_path, dementia_type)
            if os.path.exists(d_path):
                d_files = [f for f in os.listdir(d_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                random.shuffle(d_files)
                for img_name in d_files[:limit_per_class]:
                    img_path = os.path.join(d_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    img = cv2.resize(img, (img_size, img_size))
                    X.append(img.flatten()); y.append(1)

    X, y = np.array(X), np.array(y)

    # ✅ Ensure only 0 and 1 labels exist
    unique_labels = np.unique(y)
    print(f"Unique labels found: {unique_labels}")
    if not np.array_equal(unique_labels, [0, 1]):
        y = np.where(y != 0, 1, 0)  # Convert everything except 0 into 1

    print("\n✅ Image loading complete!")
    print(f"Total: {len(X)} images | Non-demented: {np.sum(y==0)} | Demented: {np.sum(y==1)}")
    return X, y


# ================================================
# STEP 4: Load Data
# ================================================
X, y = load_images_from_folder(data_root, limit_per_class=25, img_size=64)
print(f"Loaded {len(X)} total samples.")

# ================================================
# STEP 5: Preprocessing & PCA
# ================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca_components = 6
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Reduced to PCA features:", X_train_pca.shape[1])

# ================================================
# STEP 6: Quantum Feature Map & Circuit
# ================================================
n_qubits = pca_components
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.RZ(np.pi / 4, wires=0)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.state()

def quantum_kernel(X1, X2):
    K = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            psi = kernel_circuit(X1[i], X2[j])
            K[i, j] = np.abs(np.vdot(psi, psi))
    return K


# ================================================
# STEP 7: Visualize Quantum Circuit Diagram
# ================================================
example_x = np.random.random(n_qubits)
qml.draw_mpl(feature_map)(example_x)

# ================================================
# STEP 8: Compute Quantum Kernel
# ================================================
print("\n⚛️ Computing quantum kernel (please wait)...")
K_train = quantum_kernel(X_train_pca, X_train_pca)
K_test = quantum_kernel(X_test_pca, X_train_pca)

# ================================================
# STEP 9: QSVM Training
# ================================================
qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train, y_train)

# ================================================
# STEP 10: QSVM Evaluation
# ================================================
y_pred = qsvm.predict(K_test)
acc = accuracy_score(y_test, y_pred)

print("\n🧩 QSVM Results")
print("==============================")
print(f"Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["NonDemented", "Demented"]))
