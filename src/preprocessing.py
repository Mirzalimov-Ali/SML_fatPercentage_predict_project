import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.logger import get_logger

logger = get_logger('preprocessing', 'preprocessing.log')

class Preprocessing:
    def __init__(self, df):
        self.df = df.copy()
        self.encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        logger.info("Preprocessing initialized with dataset of shape: %s", self.df.shape)
    
    def fillMissingValues(self):
        try:
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    if self.df[col].dtype == 'object':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
            logger.info("Missing values filled successfully!")
            return self
        except Exception as e:
            logger.error("Error in fillMissingValues: %s", str(e))
    
    def encode(self):
        try:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    if self.df[col].nunique() <= 5:
                        dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                        self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                    else:
                        self.df[col] = self.encoder.fit_transform(self.df[col])
            logger.info("Encoding completed!")
            return self
        except Exception as e:
            logger.error("Error in encode: %s", str(e))
            raise e
    
    def scale(self):
        try:
            num_col = self.df.select_dtypes(include=['float64', 'int64']).columns.drop('fat_percentage')
            self.df[num_col] = self.scaler.fit_transform(self.df[num_col])
            logger.info("Scaling applied on %d numerical features", len(num_col))
            return self
        except Exception as e:
            logger.error("Error in scale: %s", str(e))
            raise e
    
    def logTransformation(self):
        try:
            skewness = self.df.skew()
            features_log = skewness[skewness >= 0.5].index.tolist()
            for col in features_log:
                if (self.df[col] > 0).all():
                    self.df[col] = np.log1p(self.df[col])
            logger.info("Log transformation applied on %s", features_log)
            return self
        except Exception as e:
            logger.error("Error in logTransformation: %s", str(e))
            raise e
    
    def getDataset(self):
        return self.df