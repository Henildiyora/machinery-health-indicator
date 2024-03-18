from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:

    save_model : str = os.path.join('save_models','model.pkl')

class TrainModel:

    def __init__(self):
        self.modeltrainconfig = ModelTrainerConfig()

    def train_model(self):
        pass
