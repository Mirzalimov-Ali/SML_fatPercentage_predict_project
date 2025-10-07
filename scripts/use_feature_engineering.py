import pandas as pd
from src.logger import get_logger
from src.feature_engineering import FeatureCreation
from src.preprocessing import Preprocessing
import os

logger = get_logger('use_feature_engineering', 'feature_engineering.log')

df = pd.read_csv('data/preprocessed/preprocessed_gym_dataset.csv')
logger.info(f'dataset loaded with shape: {df.shape}')

fc = FeatureCreation(df)
df = fc.create_features().getDataset()
logger.info(f'new features added')

# save dataset
output_folder = 'data/engineered'
os.makedirs('data/engineered', exist_ok=True)

output_path = os.path.join(output_folder, 'engineered_gym_dataset.csv')
logger.info(f'file created to path: {output_path}')

df.to_csv(output_path, index=False)


# ______________________________________ preprocessing (scaling, transformation) ______________________________________

preprocessing = Preprocessing(df)
df = preprocessing.scale().logTransformation().getDataset()
logger.info(f'preprocessed (scaled, transformed): {df.info}')


output_folder = 'data/final'
os.makedirs('data/final', exist_ok=True)

output_path = os.path.join(output_folder, 'final_gym_dataset.csv')
logger.info(f'file created to path: {output_path}')

df.to_csv(output_path, index=False)

print(df.head(10))