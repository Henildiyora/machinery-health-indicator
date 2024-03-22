from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object,save_json

from src.components.data_ingestion import Dataingestion

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import json
import os
import sys
import pandas as pd
import numpy as np


class CalculateDistancesTrain:

    def __init__(self,test_name,bearing_number):
        self.test_name = test_name
        self.bearing_number = bearing_number
        self.save_trainDistances = os.path.join('distances',f'{test_name}_bearing_{bearing_number}_trainDistances.json')

    def calculate_distances(self,df_train,cluster_centers,metric,scaler = False):
        '''
        calculate the distances of training data point to cluster centroids 

        Args:
            df_train : training data
            cluster_centers : cluster centroid value 
            metric : distance metric 
            scaler : bool
        '''

        try:

            logging.info(f'{metric} distance calculation of training data start')

            date = df_train['Date']
            df_train = df_train.drop('Date',axis=1)

            

            if scaler == True:

                scaler_path = f'{self.test_name}_bearing_{self.bearing_number}_scaler.pkl'

                scaler = load_object(f'save_scaler/{scaler_path}')

                df_train = scaler.transform(df_train)

            # calculate the pairwise distance for each training data points
            distances = pairwise_distances(df_train,cluster_centers,metric=metric)

            minimum_distances = np.min(distances, axis=1)

            percentile_distance_90 = np.percentile(distances,q=90,axis=0)
            percentile_distance_95 = np.percentile(distances,q=95,axis=0)
            percentile_distance_97 = np.percentile(distances,q=97,axis=0)
            percentile_distance_99 = np.percentile(distances,q=99,axis=0)

            # calculate the soft threshold 
            threshold_90 = np.mean(percentile_distance_90)
            threshold_95 = np.mean(percentile_distance_95)
            threshold_97 = np.mean(percentile_distance_97)
            threshold_99 = np.mean(percentile_distance_99)

            threshold = list((threshold_90,threshold_95,threshold_97,threshold_99))

            obj = {'train_distances':minimum_distances.tolist(),'training_dates':date.tolist(),'threshold':threshold}

            save_json(file_path=self.save_trainDistances,obj=obj)

            logging.info(f'{metric} distance calculation training data end')
        
        except Exception as e:
            raise CustomException(e,sys)











