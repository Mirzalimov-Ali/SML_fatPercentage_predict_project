import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from src.logger import get_logger

logger = get_logger('training', 'model_training.log')

class Trainer:
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y
        logger.info(f"Trainer initialized for model: {type(model).__name__}, dataset shape: {x.shape}")

    def train(self):
        try:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
            logger.info('Splitted to train/test')

            self.model.fit(self.x_train, self.y_train)
            logger.info(f"Model {type(self.model).__name__} trained successfully")

            # Embedded Feature Selection
            if hasattr(self.model, 'feature_importances_'):
                importances = pd.Series(self.model.feature_importances_, index=self.x_train.columns)
                threshold = importances.mean()

            elif hasattr(self.model, 'coef_'):
                importances = pd.Series(np.abs(self.model.coef_.ravel()), index=self.x_train.columns)
                model_name = type(self.model).__name__
                
                if model_name in ['LassoCV', 'ElasticNetCV']:
                    threshold = 0
                elif model_name in ['RidgeCV']:
                    threshold = importances.mean()
                else:
                    threshold = 0

            else:
                importances = pd.Series(np.ones(len(self.x_train.columns)), index=self.x_train.columns)
                threshold = 0

            selected_features = importances[importances > threshold].index.tolist()
            if not selected_features:
                selected_features = self.x_train.columns.tolist()
            self.selected_features = selected_features

            logger.info(f"Embedded selection done! Selected features: {self.selected_features}")

            self.x_train = self.x_train[self.selected_features]
            self.x_test = self.x_test[self.selected_features]
            self.model.fit(self.x_train, self.y_train)

            logger.info(f"Model retrained on selected features")

            return self

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")


    def evaluate(self):
        try:
            y_pred = self.model.predict(self.x_test)
            logger.info(f'Model {type(self.model).__name__} predicted successfully')

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            logger.info(f'K-Fold splited to 5 splits')
        
            self.r2 = r2_score(self.y_test, y_pred)
            self.mae = mean_absolute_error(self.y_test, y_pred)
            self.kfold = cross_val_score(self.model, self.x[self.selected_features], self.y, cv=kf, scoring='r2')

            logger.info(f"Evaluation done for model {type(self.model).__name__}: r2={self.r2}, mae={self.mae}, kfold_mean={self.kfold.mean()}, kfold_std={self.kfold.std()}")
            return self.r2, self.mae, self.kfold
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")