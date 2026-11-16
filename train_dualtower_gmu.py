import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Fix seeds for exact reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set device (CUDA GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# -------------------------------------------------------------
# 1. Load Data & Separate Multi-Modal Feature Streams
# -------------------------------------------------------------
print("📥 Loading Scaled Multimodal Dataset...")
df = pd.read_csv("data/final_fusion_ready/Scaled_Fusion_Dataset.csv")

# Feature Column Separation
sensor_cols = ['BVP', 'EDA', 'TEMP', 'ACC_Mag']
clinical_cols = ['Age', 'BMI', 'Glucose', 'Gender']

X_sensor = df[sensor_cols].values
X_clinical = df[clinical_cols].values
y = df['Label'].values

num_classes = len(np.unique(y))
print(f"Detected {num_classes} classes: {np.unique(y)}")

# Stratified Train-Test Split (80/20)
(
    X_sens_train, X_sens_test,
    X_clin_train, X_clin_test,
    y_train, y_test
) = train_test_split(
    X_sensor, X_clinical, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}\n")

# Convert to PyTorch Tensors
t_sens_train = torch.tensor(X_sens_train, dtype=torch.float32)
t_clin_train = torch.tensor(X_clin_train, dtype=torch.float32)
t_y_train = torch.tensor(y_train, dtype=torch.long)

t_sens_test = torch.tensor(X_sens_test, dtype=torch.float32).to(device)
t_clin_test = torch.tensor(X_clin_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(t_sens_train, t_clin_train, t_y_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


# -------------------------------------------------------------
# 2. Define Dual-Tower Neural Network with GMU Gating
# -------------------------------------------------------------
class DualTowerGMU(nn.Module):
    def __init__(self, sensor_dim=4, clinical_dim=4, hidden_dim=32, num_classes=8):
        super(DualTowerGMU, self).__init__()

        # Tower 1: Wearable Sensor Sub-Network (High-Frequency Dynamics)
        self.sensor_tower = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        # Tower 2: Clinical Data Sub-Network (Static Anchors)
        self.clinical_tower = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        # Gated Multimodal Unit (GMU) Layer
        # Evaluates joint representation and computes dynamic gating weights z
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Final Multiclass Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_sensor, x_clinical):
        # Forward pass through independent towers
        h_sensor = self.sensor_tower(x_sensor)
        h_clinical = self.clinical_tower(x_clinical)

        # Concatenate intermediate embeddings for gating decision
        combined = torch.cat((h_sensor, h_clinical), dim=1)
        z = self.gate_layer(combined)  # Gate weight vector between 0 and 1

        # Dynamic Feature Fusion via GMU Gating Equation
        h_fused = (z * h_sensor) + ((1 - z) * h_clinical)

        # Predict Class Logits
        logits = self.classifier(h_fused)
        return logits, z


# Initialize Model, Loss, Optimizer
model = DualTowerGMU(
    sensor_dim=len(sensor_cols),
    clinical_dim=len(clinical_cols),
    hidden_dim=32,
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# -------------------------------------------------------------
# 3. Model Training Loop
# -------------------------------------------------------------
print("=" * 60)
print("🚀 Training Dual-Tower GMU Neural Network...")
print("=" * 60)

epochs = 15
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for batch_sens, batch_clin, batch_y in train_loader:
        batch_sens = batch_sens.to(device)
        batch_clin = batch_clin.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits, _ = model(batch_sens, batch_clin)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_sens.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")

# -------------------------------------------------------------
# 4. Evaluation & Performance Benchmark
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("📊 Evaluating Dual-Tower GMU Performance...")
print("=" * 60)

model.eval()
with torch.no_grad():
    raw_logits, gate_weights = model(t_sens_test, t_clin_test)
    probs = torch.softmax(raw_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

# Compute Metrics
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average='weighted', zero_division=0)
rec = recall_score(y_test, preds, average='weighted', zero_division=0)
f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
auc = roc_auc_score(y_test, probs, multi_class='ovr')

print("\n--- Final Proposed Model Results ---")
print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}\n")

# Average Gate Weight Allocation Inspection
avg_gate_weight = gate_weights.mean(dim=0).cpu().numpy()
print(f"Gating Weight Analysis (z mean across test set): {avg_gate_weight.mean():.4f}")
print("-> Note: Values close to 1.0 favor Wearable Signals; values close to 0.0 favor Clinical Data.\n")

# Detailed Breakdown Across All 8 Classes
print("--- Per-Class Classification Report ---")
print(classification_report(y_test, preds, zero_division=0))