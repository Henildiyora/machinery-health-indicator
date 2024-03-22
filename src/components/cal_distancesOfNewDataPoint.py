from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object,save_json

from src.components.data_ingestion import Dataingestion

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


import os
import sys
import pandas as pd
import numpy as np

class CalculateDistancesNewDataPoint:

    def __init__(self,test_name,bearing_number):
        self.test_name = test_name
        self.bearing_number = bearing_number
        self.save_testDistances = os.path.join('distances',f'{test_name}_bearing_{bearing_number}_testDistances.json')

    def calculate_distance_new_datapoint(self,new_data,cluster_centers,metric,scaler=False):
        '''
        calculate the distances of testing data point to cluster centroids 

        Args:
            new_data : testing data
            cluster_centers : cluster centroid value 
            metric : distance metric 
            scaler : bool
        '''
        try:

            logging.info(f'{metric} distance calculation of new data start')

            date = new_data['Date']
            new_data = new_data.drop('Date',axis=1)

            if scaler == True:

                scaler_path = f'{self.test_name}_bearing_{self.bearing_number}_scaler.pkl'
                scaler = load_object(f'save_scaler/{scaler_path}')
                new_data = scaler.transform(new_data)

            distances = pairwise_distances(new_data,cluster_centers,metric=metric)

            min_distances = np.min(distances, axis=1)

            obj = {'test_distances':min_distances.tolist(),'testing_dates':date.tolist()}

            save_json(file_path=self.save_testDistances,obj=obj)

            logging.info(f'{metric} distance calculation of new data finish')
        
        except Exception as e:
            raise CustomException(e,sys)







        
