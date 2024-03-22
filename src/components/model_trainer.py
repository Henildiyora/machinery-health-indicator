from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.data_ingestion import Dataingestion

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import os
import sys
import pandas as pd
import numpy as np


class ModelTraining:

    '''
    train the clustering algorithem 
    '''

    def __init__(self,test_name,bearing_number):
        self.test_name = test_name
        self.bearing_number = bearing_number
        self.save_model = os.path.join('save_models',f'{test_name}_bearing_{bearing_number}_model.pkl')
        self.save_scaler = os.path.join('save_scaler',f'{test_name}_bearing_{bearing_number}_scaler.pkl')

    def model_training(self,df_train,df_test,n_cluster,scaler=None):

        '''
        function train the model 
        Args:
            df_train : training data
            df_test : testing data
            n_cluster : number of cluster 
            scaler : scaler for scale the data

        Return:
            df_label : predicted labels of training and testing data 
            cluster_centers : cluster centroids value 
        '''
        try:
            logging.info('model training start')

            score_list = []
            
            df_train = df_train.drop('Date',axis=1)
            df_test = df_test.drop('Date',axis=1)

            if scaler is not None:

                fitted_scaler = scaler.fit(df_train)

                save_object(file_path=self.save_scaler,
                            obj=fitted_scaler
                            )
                
                df_train = fitted_scaler.transform(df_train)

            for i in range(2,n_cluster):

                model = KMeans(n_clusters=i, random_state=42)

                model.fit_predict(df_train)

                score = silhouette_score(df_train,model.labels_,metric='euclidean')

                score_list.append(score)

            opt_cluster = np.argmax(score) + 2

            k_means = KMeans(n_clusters=opt_cluster,random_state=42)

            k_means.fit(df_train)

            save_object(file_path=self.save_model,
                        obj=k_means)
            
            df_train_output = k_means.predict(df_train)
            df_test_output = k_means.predict(df_test)

            df_label = np.concatenate((df_train_output,df_test_output))
            cluster_centers = k_means.cluster_centers_

            logging.info('model training end')

            return df_label,cluster_centers
        
        except Exception as e:
            raise CustomException(e,sys)


                


        
        



            


        

