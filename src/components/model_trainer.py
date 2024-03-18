from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.cluster import k_means
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:

    save_model : str = os.path.join('save_models','model.pkl')
    save_scaler :str = os.path.join('save_scaler','scaler.pkl')

class TrainModel:

    def __init__(self):
        self.modeltrainconfig = ModelTrainerConfig()

    def train_model(self,df_train,n_cluster,scaler=None):

        if scaler is not None:

            fitted_scaler = scaler.fit(df_train)

            save_object(file_path=self.modeltrainconfig.save_scaler,
                        obj=fitted_scaler
                        )
            
            X_train = fitted_scaler.transform(df_train)

            for i in range(1,n_cluster):

                


        
        



            


        

