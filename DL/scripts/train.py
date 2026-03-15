import os

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model import StructuredNN
from src.trainer import Trainer

from rich.table import Table
from rich.console import Console

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== DATA ==========
path = r'C:\SML_Projects\SML_gym_fatPercentage_predict_project\data\DL\processed'

X_train = pd.read_csv(f'{path}/X_train.csv')
X_val = pd.read_csv(f'{path}/X_val.csv')
X_test = pd.read_csv(f'{path}/X_test.csv')

y_train = pd.read_csv(f'{path}/y_train.csv')
y_val = pd.read_csv(f'{path}/y_val.csv')
y_test = pd.read_csv(f'{path}/y_test.csv')

# ========== TENSORS ==========
X_train_t = torch.from_numpy(X_train.to_numpy()).float()
X_val_t = torch.from_numpy(X_val.to_numpy()).float()
X_test_t = torch.from_numpy(X_test.to_numpy()).float()

y_train_t = torch.from_numpy(y_train.to_numpy()).float().view(-1,1)
y_val_t = torch.from_numpy(y_val.to_numpy()).float().view(-1,1)


# ========== INITIALIZATION ==========
BATCH_SIZE = 80
EPOCHS = 100
APLHA = 0.1

# ========== DATALOADER ==========
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ========== MODEL ==========
model = StructuredNN(input_dim=X_train.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=APLHA)

trainer = Trainer(model, criterion, optimizer)

train_losses, val_losses = trainer.train(
    train_loader,
    val_loader,
    epochs=EPOCHS
)


# ========== PLOT ==========
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.title("Learning Curve")
plt.show()


# ========== TEST ==========
model.eval()
with torch.no_grad():

    preds = model(X_test_t)
    y_pred = preds.numpy().flatten()


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nTest metrics")
print("R2:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

plt.scatter(y_test, y_pred)
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.title("True vs Predicted")
plt.show()


# ========== SAVE MODEL ==========
path = r'C:\SML_Projects\SML_gym_fatPercentage_predict_project\model'

os.makedirs(path, exist_ok=True)

torch.save(model.state_dict(), f"{path}/structured_nn.pth")

print("Model saved")

# ===================== RESULTS TABLE =====================
console = Console()
temp_console = Console(record=True)

table = Table(title="Neural Network Results", show_lines=True)
for col in ["Model", "R2", "MAE", "MSE", "RMSE"]:
    table.add_column(col)

table.add_row("NeuralNetwork", f"{r2:.2f}", f"{mae:.2f}", f"{mse:.2f}", f"{rmse:.2f}")
temp_console.print(table)

with open(r"C:\SML_Projects\SML_gym_fatPercentage_predict_project\results/DL_results.txt", "w", encoding="utf-8") as f:
    f.write(temp_console.export_text())