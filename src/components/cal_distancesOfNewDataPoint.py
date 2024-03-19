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

            distances_list = []

            new_data = new_data.drop('Date',axis=1)

            if scaler == True:
                scaler = load_object('save_scaler/scaler.pkl')
                new_data = scaler.transform(new_data)

            for i in range(len(new_data)):
                distances = pairwise_distances(cluster_centers,new_data[i].reshape(1,-1),metric=metric)
                distances_list.append(distances)

            # convert the list into array
            convert_to_npArray = np.array(distances_list)

            # convert the 3d array into 2d 
            all_elements_2d = convert_to_npArray[:, :, 0]
            min_distances = np.min(all_elements_2d, axis=1)

            # print(f'min distances = \n{min_distances}')

            logging.info(f'{metric} distance calculation of new data finish')

            return min_distances
        
        except Exception as e:
            raise CustomException(e,sys)







        
