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

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:

    save_model : str = os.path.join('save_models','model.pkl')
    save_scaler :str = os.path.join('save_scaler','scaler.pkl')

class TrainModel:

    def __init__(self):
        self.modeltrainconfig = ModelTrainerConfig()

    def train_model(self,df_train,df_test,n_cluster,scaler=None):

        logging.info('model training start')

        score_list = []
        
        df_train = df_train.drop('Date',axis=1)
        df_test = df_test.drop('Date',axis=1)

        if scaler is not None:

            fitted_scaler = scaler.fit(df_train)

            save_object(file_path=self.modeltrainconfig.save_scaler,
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

        save_object(file_path=self.modeltrainconfig.save_model,
                    obj=k_means)
        
        df_train_output = k_means.predict(df_train)
        df_test_output = k_means.predict(df_test)

        df_label = np.concatenate((df_train_output,df_test_output))
        cluster_centers = k_means.cluster_centers_

        logging.info('model training end')

        return df_label,cluster_centers

# if __name__ == "__main__":

#     dataingestion = Dataingestion()
#     df_train,df_test,date = dataingestion.initiate_data_ingestion(train_date='2004.02.15.13.12.39')

#     trainmodel = TrainModel()
#     scaler = StandardScaler()
#     trainmodel.train_model(df_train=df_train,df_test=df_test,n_cluster=5,scaler=scaler)


                


        
        



            


        

