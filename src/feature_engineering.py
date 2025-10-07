import pandas as pd
from src.logger import get_logger

logger = get_logger("feature_engineering", "feature_engineering.log")

class FeatureCreation:
    def __init__(self, df):
        self.df = df.copy()
        logger.info(f"FeatureCreation initialized with dataset shape: {self.df.shape}")

    def create_features(self):
        try:
            self.df['bmi'] = self.df['weight'] / ((self.df['height'] / 100) ** 2)
            logger.info("Feature 'bmi' created")

            self.df['hrr'] = self.df['max_bpm'] - self.df['resting_bpm']
            logger.info("Feature 'hrr' created")

            self.df['intensity'] = (self.df['avg_bpm'] - self.df['resting_bpm']) / self.df['hrr']
            logger.info("Feature 'intensity' created")

            self.df['calories_per_min'] = self.df['calories_burned'] / self.df['session_duration']
            logger.info("Feature 'calories_per_min' created")

            self.df['water_per_session'] = self.df['water_intake'] / self.df['workout_frequency']
            logger.info("Feature 'water_per_session' created")

            self.df['experience_intensity_ratio'] = self.df['experience_level'] * self.df['intensity']
            logger.info("Feature 'experience_intensity_ratio' created")

            logger.info(f"All features created successfully. Dataset shape now: {self.df.shape}")

            return self

        except Exception as e:
            logger.error(f"Error in create_features: {e}")

    def getDataset(self):
        return self.df