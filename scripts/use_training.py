import os
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
from src.logger import get_logger
from rich.table import Table
from rich.console import Console

# ===================== PATH =====================
os.chdir(r'C:\SML_Projects\SML_gym_fatPercentage_predict_project')

logger = get_logger('use_training', 'training.log')

# ===================== DATA LOAD =====================
x_train = pd.read_csv('data/preprocessed/preprocessed_x_train.csv')
x_test  = pd.read_csv('data/preprocessed/preprocessed_x_test.csv')

y_train = pd.read_csv('data/split/y_train.csv').values.ravel()
y_test  = pd.read_csv('data/split/y_test.csv').values.ravel()

logger.info("Data loaded successfully")

kf = KFold(n_splits=3, shuffle=True, random_state=42)

# ===================== FIT =====================
model = AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(x_train, y_train)

logger.info("Best AdaBoostRegressor trained")

# ===================== EVALUATION =====================
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

cv_scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

kf_mean = cv_scores.mean()
kf_std = cv_scores.std()

# ===================== SAVE PIPELINE =====================
os.makedirs('pipeline', exist_ok=True)
dump(model, 'pipeline/final_pipeline.joblib', compress=3)

logger.info("Final regression model saved")

# ===================== RESULTS TABLE =====================
console = Console()
temp_console = Console(record=True)

table = Table(title="AdaBoost Results", show_lines=True)
for col in ["Algorithm", "R2", "MAE", "MSE", "RMSE", "K-Fold mean", "K-Fold std"]:
    table.add_column(col)

table.add_row("AdaBoost", f"{r2:.2f}", f"{mae:.2f}", f"{mse:.2f}", f"{rmse:.2f}", f"{kf_mean:.2f}", f"{kf_std:.2f}")
temp_console.print(table)

with open("results/final_results.txt", "w", encoding="utf-8") as f:
    f.write(temp_console.export_text())
logger.info("Comparison table saved at results/final_results.txt")

print("Results saved to results/final_results.txt")