import pickle
import pandas as pd
import numpy as np
import os
import warnings

# Force-ignore the NumPy 2.4 internal pickle warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*align.*")

subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
base_path = 'data/WESAD/'
output_path = 'data/'


def extract_all_9_prep(sub_id):
    file_path = os.path.join(base_path, f'{sub_id}/{sub_id}.pkl')
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Wrist Sensors
    w = data['signal']['wrist']

    # 1. BVP (64Hz -> 4Hz)
    bvp = w['BVP'][::16]

    # 2. ACC (32Hz -> 4Hz) + Magnitude
    acc = w['ACC'][::8]
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))

    # 3. EDA (4Hz)
    eda = w['EDA']

    # 4. TEMP (4Hz)
    temp = w['TEMP']

    # 5. Label (700Hz -> 4Hz)
    # 700 / 4 = 175. This is critical for alignment!
    lbl = data['label'][::175]

    # Sync all 5 wearable columns
    min_l = min(len(bvp), len(eda), len(temp), len(acc_mag), len(lbl))

    return pd.DataFrame({
        'BVP': bvp[:min_l].flatten(),
        'EDA': eda[:min_l].flatten(),
        'TEMP': temp[:min_l].flatten(),
        'ACC_Mag': acc_mag[:min_l],
        'Label': lbl[:min_l]
    })


if __name__ == "__main__":
    print("🚀 Running deep extraction for 9-column requirement...")
    for sub in subjects:
        df = extract_all_9_prep(sub)
        if df is not None:
            df.to_csv(f'data/{sub}_cleaned.csv', index=False)
            print(f"✅ {sub}_cleaned.csv created (5 columns ready)")

    print("\nNext: Run your mimicGenerator.py to add the 4 clinical columns!")