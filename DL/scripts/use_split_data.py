import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ========= LOAD DATA =========
df = pd.read_csv(r"C:\SML_Projects\SML_gym_fatPercentage_predict_project\data\raw\gym_dataset.csv")

# ========= TARGET =========
X = df.drop("fat_percentage", axis=1)
y = df["fat_percentage"]

# ========= SPLIT =========
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1765, random_state=42)

# ========= SAVE =========

PATH = r'C:\SML_Projects\SML_gym_fatPercentage_predict_project\data'

os.makedirs(f"{PATH}/DL/split", exist_ok=True)

X_train.to_csv(f"{PATH}/DL/split/X_train.csv", index=False)
X_val.to_csv(f"{PATH}/DL/split/X_val.csv", index=False)
X_test.to_csv(f"{PATH}/DL/split/X_test.csv", index=False)

y_train.to_csv(f"{PATH}/DL/split/y_train.csv", index=False)
y_val.to_csv(f"{PATH}/DL/split/y_val.csv", index=False)
y_test.to_csv(f"{PATH}/DL/split/y_test.csv", index=False)

print("Data successfully split and saved.")