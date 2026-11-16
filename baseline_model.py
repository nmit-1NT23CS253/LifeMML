import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Fix random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------------------------------------
# 1. Load Preprocessed Data
# -------------------------------------------------------------
print("Loading dataset...")
data = pd.read_csv("data/final_fusion_ready/Scaled_Fusion_Dataset.csv")

feature_cols = ['BVP', 'EDA', 'TEMP', 'ACC_Mag', 'Age', 'BMI', 'Glucose', 'Gender']
X = data[feature_cols].values
y = data['Label'].values

classes = np.unique(y)
num_classes = len(classes)
print(f"Dataset loaded successfully. Found {num_classes} classes: {classes}")

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}\n")


# -------------------------------------------------------------
# 2. Baseline 1: Logistic Regression
# -------------------------------------------------------------
print("=" * 50)
print("Running Baseline 1: Logistic Regression")
print("=" * 50)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

lr_preds = log_reg.predict(X_test)
lr_probs = log_reg.predict_proba(X_test)

lr_acc = accuracy_score(y_test, lr_preds)
lr_prec = precision_score(y_test, lr_preds, average='weighted', zero_division=0)
lr_rec = recall_score(y_test, lr_preds, average='weighted', zero_division=0)
lr_f1 = f1_score(y_test, lr_preds, average='weighted', zero_division=0)
lr_auc = roc_auc_score(y_test, lr_probs, multi_class='ovr')

print(f"Accuracy : {lr_acc:.4f}")
print(f"Precision: {lr_prec:.4f}")
print(f"Recall   : {lr_rec:.4f}")
print(f"F1-Score : {lr_f1:.4f}")
print(f"AUC-ROC  : {lr_auc:.4f}\n")


# -------------------------------------------------------------
# 3. Baseline 2: Early Fusion MLP (PyTorch)
# -------------------------------------------------------------
print("=" * 50)
print("Running Baseline 2: Early Fusion MLP")
print("=" * 50)

# Prepare tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


class SimpleMLP(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


model = SimpleMLP(in_features=X.shape[1], num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    raw_outputs = model(X_test_t)
    mlp_probs = torch.softmax(raw_outputs, dim=1).numpy()
    mlp_preds = np.argmax(mlp_probs, axis=1)

mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_prec = precision_score(y_test, mlp_preds, average='weighted', zero_division=0)
mlp_rec = recall_score(y_test, mlp_preds, average='weighted', zero_division=0)
mlp_f1 = f1_score(y_test, mlp_preds, average='weighted', zero_division=0)
mlp_auc = roc_auc_score(y_test, mlp_probs, multi_class='ovr')

print("\n--- MLP Performance ---")
print(f"Accuracy : {mlp_acc:.4f}")
print(f"Precision: {mlp_prec:.4f}")
print(f"Recall   : {mlp_rec:.4f}")
print(f"F1-Score : {mlp_f1:.4f}")
print(f"AUC-ROC  : {mlp_auc:.4f}")