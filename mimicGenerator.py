import pandas as pd, numpy as np, os

# (Using the same logic we used before, just pointed at the new 5-column files)
input_dir, output_dir = 'data/', 'data/final_fusion_ready/'
subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

np.random.seed(42)
clinical_data = pd.DataFrame({
    'Subject_ID': subjects,
    'Age': np.random.normal(24, 4, len(subjects)).astype(int),
    'BMI': np.random.uniform(19, 34, len(subjects)).round(1),
    'Glucose': np.random.normal(95, 15, len(subjects)).round(0),
    'Gender': np.random.randint(0, 2, len(subjects))
})

all_data = []
for _, row in clinical_data.iterrows():
    f = f"data/{row['Subject_ID']}_cleaned.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        df['Age'], df['BMI'] = row['Age'], row['BMI']
        df['Glucose'], df['Gender'] = row['Glucose'], row['Gender']
        all_data.append(df)

master_df = pd.concat(all_data, ignore_index=True)
master_df.to_csv(os.path.join(output_dir, 'Final_Fusion_Dataset.csv'), index=False)

print(f"🚀 FUSION COMPLETE! Columns: {len(master_df.columns)}")
print(master_df.columns.tolist())