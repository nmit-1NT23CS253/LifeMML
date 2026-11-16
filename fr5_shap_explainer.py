import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your trained Dual-Tower model and test dataloaders are ready
# model = DualTowerGMU(...)
# model.load_state_dict(torch.load("model_checkpoint.pth"))
# model.eval()

def model_predict_wrapper(x_numpy):
    """Wrapper function for SHAP KernelExplainer to accept NumPy arrays
       and return model output probabilities for classification."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        # Split your input features back into sensor and clinical tensors if needed by your forward pass
        # sensor_feat = x_tensor[:, :4]
        # clinical_feat = x_tensor[:, 4:]
        outputs = model(x_tensor) # Adjust based on your model's forward method
        probabilities = torch.softmax(outputs, dim=1)
    return probabilities.numpy()

# 1. Select a background dataset (e.g., 100 samples from your training set for baseline reference)
# background_data = X_train.sample(100, random_state=42).values
# test_samples = X_test.sample(50, random_state=42).values

# 2. Initialize KernelExplainer
# explainer = shap.KernelExplainer(model_predict_wrapper, background_data)

# 3. Compute SHAP values for test samples
# shap_values = explainer.shap_values(test_samples)

# 4. Generate Visualizations for your Report
# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values[1], test_samples, feature_names=['BVP', 'EDA', 'TEMP', 'ACC_Mag', 'Age', 'BMI', 'Glucose', 'Gender'], show=False)
# plt.title("SHAP Feature Importance Summary for Lifestyle Disorder Prediction")
# plt.tight_layout()
# plt.savefig("shap_summary_plot.png", dpi=300)
# plt.show()