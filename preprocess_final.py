import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_multimodal_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"🚨 Error: Master dataset not found at {input_file}")
        return

    # 1. Load the unified 9-column dataset
    print("📖 Loading Final_Fusion_Dataset.csv...")
    df = pd.read_csv(input_file)
    print(f"Original Shape: {df.shape}")

    # 2. Separate columns by data type
    # Continuous features that NEED scaling due to wildly different units (e.g., °C vs mg/dL)
    continuous_features = ['BVP', 'EDA', 'TEMP', 'ACC_Mag', 'Age', 'BMI', 'Glucose']

    # Categorical/Target features that MUST NOT be scaled
    passthrough_features = ['Gender', 'Label']

    print(f"⚖️ Normalizing continuous features to [0, 1] range: {continuous_features}")

    # 3. Apply Min-Max Scaling
    scaler = MinMaxScaler()

    # Create a copy to avoid altering the original dataframe in memory
    scaled_df = df.copy()
    scaled_df[continuous_features] = scaler.fit_transform(df[continuous_features])

    # 4. Ensure the columns are in the exact structural order needed for the towers
    column_order = ['BVP', 'EDA', 'TEMP', 'ACC_Mag', 'Label', 'Age', 'BMI', 'Glucose', 'Gender']
    scaled_df = scaled_df[column_order]

    # 5. Save the ready-to-train dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    scaled_df.to_csv(output_file, index=False)

    print("\n🚀 PREPROCESSING & SCALING COMPLETE!")
    print(f"Saved to: {output_file}")
    print("\n--- Sample Row Verification (Scaled) ---")
    print(scaled_df.head(1).to_string(index=False))


if __name__ == "__main__":
    # Define paths matching your PyCharm structure
    INPUT_PATH = 'data/final_fusion_ready/Final_Fusion_Dataset.csv'
    OUTPUT_PATH = 'data/final_fusion_ready/Scaled_Fusion_Dataset.csv'

    scale_multimodal_dataset(INPUT_PATH, OUTPUT_PATH)