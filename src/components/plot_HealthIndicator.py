from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object
from src.components.data_ingestion import Dataingestion
from src.components.model_trainer import TrainModel
from src.components.cal_distancesOfTrainDataPoint import CalculateDistancesTrain
from src.components.cal_distancesOfNewDataPoint import CalculateDistancesNewDataPoint

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass


@dataclass
class PlotHealthIndicatorConfig:

    save_figure : str = os.path.join('save_healthIndicator',f'healthindicator.png')

class PlotHealthIndicator:

    def __init__(self):
        plotHealthIndicatorConfig = PlotHealthIndicatorConfig()

    def plot_health_indicator(self,train_distances,test_distances,percentile_threshold,date):
        """
        Plot the max distances for each data point along with the mean distance on a dashed red line.

         Args:
             max_distances (list): Max distance for each data point from all cluster centroids
             mean_distance_train (float): Mean distance calculated during training
             timestamp (list): List of timestamps corresponding to data points
         """

        cluster_distances = np.concatenate((train_distances,test_distances))

        colors = ["r" if status > percentile_threshold else "g" for status in cluster_distances]

        plt.figure(figsize=(10, 8))
        plt.style.use('ggplot')
        plt.scatter(range(len(cluster_distances)),cluster_distances,c = colors,s=4,alpha=0.7,label='Min Distances')
        plt.plot(cluster_distances,color='gray',alpha=0.3, linestyle='-',linewidth=0.5)
        # plt.hlines(y=minimum_threshold, xmin=0, xmax=len(cluster_distances), colors='green', linestyles='dashed', label='min Threshold')
        plt.hlines(y=percentile_threshold, xmin=0, xmax=len(cluster_distances), colors='blue', linestyles='dashed', label='Percentile Threshold')
        plt.xticks(np.arange(0, len(date), 100), date[::100], rotation=90, size=8)
        plt.title(f'Euclidean Distances for Each Data Point')
        plt.xlabel('Timestamp')
        plt.ylabel('Euclidean Distances')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# if __name__ == "__main__":

#     dataingestion = Dataingestion()
#     df_train,df_test,date = dataingestion.initiate_data_ingestion(train_date='2004.02.15.13.12.39')

#     trainmodel = TrainModel()
#     scaler = MinMaxScaler()
#     df_label,cluster_centers = trainmodel.train_model(df_train=df_train,df_test=df_test,n_cluster=5,scaler=scaler)

#     cal_train_distance = CalculateDistancesTrain()
#     metric = 'euclidean'
#     minimum_distances,threshold = cal_train_distance.calculate_distances(df_train=df_train,cluster_centers=cluster_centers,metric=metric,scaler=True)
    
#     cal_new_datapoint_distance = CalculateDistancesNewDataPoint()
#     new_data_min_distance = cal_new_datapoint_distance.calculate_distance_new_datapoint(new_data=df_test,cluster_centers=cluster_centers,metric=metric,scaler=True)


#     plot = PlotHealthIndicator()
#     plot.plot_health_indicator(minimum_distances,new_data_min_distance,threshold,date)
