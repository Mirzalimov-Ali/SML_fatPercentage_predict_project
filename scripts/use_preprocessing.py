import pandas as pd
from src.logger import get_logger
from src.preprocessing import Preprocessing
import os

logger = get_logger('use_preprocessing', 'preprocessing.log')

df = pd.read_csv('data/raw/gym_dataset.csv')
logger.info(f'dataset loaded with shape: {df.shape}')

preprocessing = Preprocessing(df)
df = preprocessing.fillMissingValues().encode().getDataset()
logger.info(f'data has preprocessed')

# save dataset
output_folder = 'data/preprocessed'
os.makedirs('data/preprocessed', exist_ok=True)

output_path = os.path.join(output_folder, 'preprocessed_gym_dataset.csv')
logger.info(f'file created to path: {output_path}')

df.to_csv(output_path, index=False)

print(df.head(10))