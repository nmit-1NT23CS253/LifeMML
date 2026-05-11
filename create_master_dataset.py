import pandas as pd
import os

# List of all subjects you just cleaned
subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

all_data = []

# This dictionary simulates the "MIMIC-IV" Clinical Profile for each person
# In a real scenario, this would come from your MIMIC lab results
clinical_profiles = {
    'S2': {'age': 25, 'bmi': 22.5, 'glucose': 85},
    'S3': {'age': 28, 'bmi': 27.1, 'glucose': 105},  # Pre-diabetic markers
    'S4': {'age': 24, 'bmi': 21.0, 'glucose': 90},
    # ... you can randomize or fill others similarly
}

for sub in subjects:
    file_path = f'data/{sub}_cleaned.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Add the Clinical Features (The "Fusion" prep)
        profile = clinical_profiles.get(sub, {'age': 25, 'bmi': 24.0, 'glucose': 95})  # Default values
        df['age'] = profile['age']
        df['bmi'] = profile['bmi']
        df['glucose'] = profile['glucose']
        df['subject_id'] = sub

        all_data.append(df)

# Merge everything into ONE giant file
master_df = pd.concat(all_data, ignore_index=True)

# Important: Remove Label 0 (Undefined) and Label 3 (Amusement)
# We only want 1 (Baseline) and 2 (Stress/Risk) for binary classification
master_df = master_df[master_df['Label'].isin([1, 2])]

# Save the final masterpiece
master_df.to_csv('data/master_multimodal_data.csv', index=False)
print(f"Master Dataset Created! Total Rows: {len(master_df)}")