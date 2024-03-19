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

            distances_list = []
            percentile_distances = []
            max_distances = []
            min_distances = []

            if scaler == True:

                scaler = load_object('save_scaler/scaler.pkl')

                df_train = scaler.transform(df_train)

            # calculate the pairwise distance for each training data points
            for i in range(len(df_train)):
                distances = pairwise_distances(cluster_centers,df_train[i].reshape(1,-1),metric=metric)
                distances_list.append(distances)

            # convert the list into array
            convert_to_npArray = np.array(distances_list)

            # convert the 3d array into 2d 
            all_elements_2d = convert_to_npArray[:, :, 0]
            train_minimum_distances = np.min(all_elements_2d, axis=1)

            # store the each cluster distance into dict 
            for i in range(len(cluster_centers)):

                iter_values = list(all_elements_2d[:,i])

                max_distance = np.max(iter_values)
                min_distance = np.min(iter_values)
                percentile_distance = np.percentile(iter_values,99.7)

                min_distances.append(min_distance)
                max_distances.append(max_distance)
                percentile_distances.append(percentile_distance)

            logging.info(f'{metric} distance calculation training data end')

            return min_distances,max_distances,percentile_distances,train_minimum_distances
        
        except Exception as e:
            raise CustomException(e,sys)









