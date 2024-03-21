from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object

from src.components.data_ingestion import Dataingestion

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


import os
import sys
import pandas as pd
import numpy as np


class CalculateDistancesTrain:

    def calculate_distances(self,df_train,cluster_centers,metric,scaler = False):

        try:

            logging.info(f'{metric} distance calculation of training data start')

            df_train = df_train.drop('Date',axis=1)

            if scaler == True:

                scaler = load_object('save_scaler/scaler.pkl')

                df_train = scaler.transform(df_train)

            # calculate the pairwise distance for each training data points
            distances = pairwise_distances(df_train,cluster_centers,metric=metric)

            minimum_distances = np.min(distances, axis=1)

            percentile_distance = np.percentile(distances,q=99.7,axis=0)

            # calculate the soft threshold 
            threshold = np.mean(percentile_distance)

            logging.info(f'{metric} distance calculation training data end')

            return minimum_distances,threshold
        
        except Exception as e:
            raise CustomException(e,sys)









