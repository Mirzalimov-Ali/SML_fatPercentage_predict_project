import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV


from tabulate import tabulate
from rich.console import Console
from rich.table import Table


from joblib import dump
import os
from src.training import Trainer

from src.logger import get_logger

logger = get_logger('use_train', 'model_training.log')

# ____________________________________________ Single Split __________________________________________________

df = pd.read_csv('data/final/final_gym_dataset.csv')

try:
    x = df.drop('fat_percentage', axis=1)
    y = df['fat_percentage']

    logger.info(f"Dataset loaded with shape {df.shape}")

except Exception as e:
    logger.error(f'Error during single split: {str(e)}')

# _____________________________________________ train model __________________________________________________

try:
    models = [
        LinearRegression(),
        LassoCV(cv=10, random_state=42),
        RidgeCV(cv=10),
        ElasticNetCV(cv=10, random_state=42),
        DecisionTreeRegressor(random_state=42),
        RandomForestRegressor(random_state=42, n_estimators=200),
        GradientBoostingRegressor(random_state=42),
        HistGradientBoostingRegressor(random_state=42),
        ExtraTreesRegressor(random_state=42),
        AdaBoostRegressor(random_state=42),
        xgboost.XGBRegressor(random_state=42),
        SVR(kernel='rbf', C=20.0),
        KNeighborsRegressor(n_neighbors=10)
    ]

    results = []
    best_r2 = -float("inf") # eng kichkina mumkun bulgan infinity son
    best_trained_model = None   # object
    best_model_name = ""    # joblibga save qilyotkanda modelni name

    for model in models:
        trainer = Trainer(model, x, y)
        trainer.train().evaluate()

        results.append([model.__class__.__name__, trainer.r2, trainer.mae, trainer.kfold.mean(), trainer.kfold.std()])

        dump(trainer.model, f'model/{model.__class__.__name__}.joblib')

        if trainer.r2 > best_r2:
            best_r2 = trainer.r2
            best_trained_model = trainer.model
            best_model_name = model.__class__.__name__

except Exception as e:
    logger.error(f'Error during train all models: {str(e)}')

# ______________________________________________ compare table ____________________________________________________

console = Console()

results_sorted = sorted(results, key=lambda i: i[1], reverse=True) # r2 buyicha sort qilish

best_model = results_sorted[0]
worst_model = results_sorted[-1]

table = Table(title="Model Compare", show_lines=True)

table.add_column("Algorithm")
table.add_column("R2 score")
table.add_column("Mean Absolute Error", justify="right")
table.add_column("K-Fold Mean", justify="right")
table.add_column("K-Fold Std", justify="right")

try:
    for row in results_sorted:
        algo, r2, mae, kmean, kstd = row

        if row == best_model:
            table.add_row(
                f"[bold green]{algo}[/bold green]",
                f"[bold green]{r2:.6f}[/bold green]",
                f"[bold green]{mae:.6f}[/bold green]",
                f"[bold green]{kmean:.6f}[/bold green]",
                f"[bold green]{kstd:.6f}[/bold green]"
            )
        elif row == worst_model:
            table.add_row(
                f"[bold red]{algo}[/bold red]",
                f"[bold red]{r2:.6f}[/bold red]",
                f"[bold red]{mae:.6f}[/bold red]",
                f"[bold red]{kmean:.6f}[/bold red]",
                f"[bold red]{kstd:.6f}[/bold red]"
            )
        else:
            table.add_row(algo, f"{r2:.6f}", f"{mae:.6f}", f"{kmean:.6f}", f"{kstd:.6f}")
            
    logger.info(f"\n{table}")

except Exception as e:
    logger.error(f'Error during create table: {str(e)}')

# ___________________________________________ save comparison _________________________________________________

try:
    def SaveComparison(table):
        temp_console = Console(record=True)   # vaqtincha console (shu console orqali chiqargan hammanarsa memoryda save buladi)
        temp_console.print(table)             # aynan tableni chiqaramiz
        text = temp_console.export_text()     # table object bulgani uchun buni text kurinishga apkelamiz
        with open('results/all_model_compare.txt', 'w', encoding='utf-8') as f:
            f.write(text)

    SaveComparison(table)
    logger.info(f"Comparison table saved at results/all_model_compare.txt")

    table_log = tabulate(results_sorted, headers=['Algorithm', 'R2', 'MAE', 'K-Fold Mean', 'K-Fold Std'], tablefmt='grid', floatfmt='.3f')
    logger.info(f'\n{table_log}')

except Exception as e:
    logger.error(f'Error during save table comparison of all models: {str(e)}')

# ___________________________________________ save best model _________________________________________________

if best_trained_model is not None:
    os.makedirs('model/best', exist_ok=True)
    save_path = f'model/best/{best_model_name}.joblib'
    dump(best_trained_model, save_path)
    logger.info(f"Best model '{best_model_name}' saved successfully at '{save_path}' with r2={best_r2}")
else:
    logger.error("No model was selected to save. Check the training loop.")