import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib


# ============ DATA LOAD ============
path = r'C:\SML_Projects\SML_gym_fatPercentage_predict_project\data\DL\split'

X_train = pd.read_csv(f'{path}/X_train.csv')
X_val = pd.read_csv(f'{path}/X_val.csv')
X_test = pd.read_csv(f'{path}/X_test.csv')

y_train = pd.read_csv(f'{path}/y_train.csv')
y_val = pd.read_csv(f'{path}/y_val.csv')
y_test = pd.read_csv(f'{path}/y_test.csv')


# ============ CATEGORICAL ENCODING ============
cat_cols = X_train.select_dtypes(include=['object']).columns

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

if len(cat_cols) > 0:
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = encoder.transform(X_val[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])


# ============ SCALING ============
scaler = StandardScaler()

columns = X_train.columns

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=columns)


# ============ SAVE PREPROCESSED DATA ============
PATH = r'C:\SML_Projects\SML_gym_fatPercentage_predict_project\data'

os.makedirs(f"{PATH}/DL/processed", exist_ok=True)

pd.DataFrame(X_train).to_csv(f"{PATH}/DL/processed/X_train.csv", index=False)
pd.DataFrame(X_val).to_csv(f"{PATH}/DL/processed/X_val.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{PATH}/DL/processed/X_test.csv", index=False)

y_train.to_csv(f"{PATH}/DL/processed/y_train.csv", index=False)
y_val.to_csv(f"{PATH}/DL/processed/y_val.csv", index=False)
y_test.to_csv(f"{PATH}/DL/processed/y_test.csv", index=False)

print(f'datasets saved!')


# ============ SAVE ENCODER & SCALER ============
joblib.dump(scaler, f"{PATH}/DL/processed/scaler.joblib")
joblib.dump(encoder, f"{PATH}/DL/processed/encoder.joblib")