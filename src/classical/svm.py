# ===============================
# STEP 1: Mount Google Drive
# ===============================
from google.colab import drive
drive.mount('/content/drive')

# ===============================
# STEP 2: Define Dataset Path
# ===============================
# Change this path according to your Google Drive folder
dataset_path = "/content/drive/MyDrive/Dataset_1/AD_Vs_CN"

# Structure should look like:
# AD vs CN/
#   ├── Moderate Demented/   (AD)
#   └── Non Demented/        (CN)

# ===============================
# STEP 3: Import Libraries
# ===============================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===============================
# STEP 4: Preprocessing Generator
# ===============================
IMG_SIZE = (224, 224)   # Standard for ResNet50
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # ResNet50 specific preprocessing
    validation_split=0.2   # 80% train, 20% validation
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',   # binary labels (0,1)
    subset='training',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class Indices:", train_generator.class_indices)  # e.g., {'Moderate Demented':0, 'Non Demented':1}

# ===============================
# STEP 5: Load Pretrained ResNet50
# ===============================
# Load ResNet50 without top layer (only feature extractor)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# ===============================
# STEP 6: Extract Features
# ===============================
def extract_features(generator, model):
    features = model.predict(generator, verbose=1)
    labels = generator.classes
    return features, labels

train_features, train_labels = extract_features(train_generator, base_model)
val_features, val_labels = extract_features(val_generator, base_model)

print("Train Features Shape:", train_features.shape)  # e.g. (num_samples, 2048)
print("Validation Features Shape:", val_features.shape)

# ===============================
# STEP 7: Save Features to CSV
# ===============================
# Train Set
train_df = pd.DataFrame(train_features)
train_df['label'] = train_labels
train_df.to_csv("/content/drive/MyDrive/alzheimer_train_features.csv", index=False)

# Validation Set
val_df = pd.DataFrame(val_features)
val_df['label'] = val_labels
val_df.to_csv("/content/drive/MyDrive/alzheimer_val_features.csv", index=False)

print("✅ Features saved as CSV in Google Drive")


# ===============================
# STEP 1: Import Libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===============================
# STEP 2: Load Extracted Features
# ===============================
train_df = pd.read_csv("/content/drive/MyDrive/alzheimer_train_features.csv")
val_df   = pd.read_csv("/content/drive/MyDrive/alzheimer_val_features.csv")

# Features & Labels
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_val = val_df.drop('label', axis=1).values
y_val = val_df['label'].values

print("Train Shape:", X_train.shape, y_train.shape)
print("Validation Shape:", X_val.shape, y_val.shape)

# ===============================
# STEP 3: Feature Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ===============================
# STEP 4: Train Different Models
# ===============================

# ---- Model 1: SVM ----
svm_clf = SVC(kernel='rbf', C=1, probability=True, random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_val)
print("🔹 SVM Results:")
print(classification_report(y_val, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_svm))

# ---- Model 2: Random Forest ----
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_val)
print("\n🔹 Random Forest Results:")
print(classification_report(y_val, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_rf))

# ---- Model 3: XGBoost ----
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_val)
print("\n🔹 XGBoost Results:")
print(classification_report(y_val, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_xgb))

# ===============================
# STEP 5: Compare Accuracies
# ===============================
print("\n✅ Accuracy Scores:")
print("SVM Accuracy:", accuracy_score(y_val, y_pred_svm))
print("RF Accuracy:", accuracy_score(y_val, y_pred_rf))
print("XGBoost Accuracy:", accuracy_score(y_val, y_pred_xgb))
