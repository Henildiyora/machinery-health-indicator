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

class CalculateDistancesNewDataPoint:

    def calculate_distance_new_datapoint(self,new_data,cluster_centers,metric,scaler=False):

        try:

            logging.info(f'{metric} distance calculation of new data start')

            new_data = new_data.drop('Date',axis=1)

            if scaler == True:
                scaler = load_object('save_scaler/scaler.pkl')
                new_data = scaler.transform(new_data)

            distances = pairwise_distances(new_data,cluster_centers,metric=metric)

            min_distances = np.min(distances, axis=1)

            logging.info(f'{metric} distance calculation of new data finish')

            return min_distances
        
        except Exception as e:
            raise CustomException(e,sys)







        
