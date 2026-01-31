import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import get_logger

logger = get_logger('feature_engineering', 'feature_engineering.log')


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # ---------------- CARDIO ----------------
        X['hrr'] = X['max_bpm'] - X['resting_bpm']
        logger.info("Feature 'hrr' created")

        X['intensity'] = (X['avg_bpm'] - X['resting_bpm']) / X['hrr']
        logger.info("Feature 'intensity' created")

        X['cardio_load'] = X['intensity'] * X['session_duration']
        logger.info("Feature 'cardio_load' created")

        X['weekly_cardio_load'] = X['cardio_load'] * X['workout_frequency']
        logger.info("Feature 'weekly_cardio_load' created")

        X['bpm_efficiency'] = X['calories_burned'] / X['avg_bpm']
        logger.info("Feature 'bpm_efficiency' created")

        # ---------------- CALORIES ----------------
        X['calories_per_min'] = X['calories_burned'] / X['session_duration']
        logger.info("Feature 'calories_per_min' created")

        X['calories_per_kg'] = X['calories_burned'] / X['weight']
        logger.info("Feature 'calories_per_kg' created")

        X['fat_burn_efficiency'] = X['calories_per_min'] * X['intensity']
        logger.info("Feature 'fat_burn_efficiency' created")

        # ---------------- BODY ----------------
        X['weight_height_ratio'] = X['weight'] / X['height']
        logger.info("Feature 'weight_height_ratio' created")

        X['bmi_age_interaction'] = X['bmi'] * X['age']
        logger.info("Feature 'bmi_age_interaction' created")

        X['weight_age_ratio'] = X['weight'] / X['age']
        logger.info("Feature 'weight_age_ratio' created")

        X['fat_risk_score'] = X['bmi'] * X['weight'] / X['height']
        logger.info("Feature 'fat_risk_score' created")

        # ---------------- WATER ----------------
        X['water_per_session'] = X['water_intake'] / X['workout_frequency']
        logger.info("Feature 'water_per_session' created")

        X['water_per_kg'] = X['water_intake'] / X['weight']
        logger.info("Feature 'water_per_kg' created")

        X['hydration_score'] = X['water_per_session'] / X['session_duration']
        logger.info("Feature 'hydration_score' created")

        # ---------------- EXPERIENCE ----------------
        X['experience_intensity_ratio'] = X['experience_level'] * X['intensity']
        logger.info("Feature 'experience_intensity_ratio' created")

        X['experience_load'] = X['experience_level'] * X['cardio_load']
        logger.info("Feature 'experience_load' created")

        X['experience_frequency_score'] = X['experience_level'] * X['workout_frequency']
        logger.info("Feature 'experience_frequency_score' created")

        logger.info(f"All features created successfully. Dataset shape now: {X.shape}")

        return X

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        return state
