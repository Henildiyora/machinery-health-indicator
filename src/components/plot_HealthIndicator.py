from src.exception import CustomException
from src.logger import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


class PlotHealthIndicator:

    def __init__(self,test_name,bearing_number):
        self.test_name = test_name,
        self.bearing_number = bearing_number
        self.save_plot = os.path.join('saved_healthIndicator_plots',f'{test_name}_bearing_{bearing_number}_healthindicator.png')

    def plot_health_indicator(self,train_distances,test_distances,threshold_list,date):
        """
        Plot the distances for each data point along with the threshold.

         Args:
             train_distances (list): calculated train distances
             test_distances (list): calculated test distances
             threshold (int) : calculated threshold value
             date (list): List of timestamps corresponding to data points
         """
        
        try:
            logging.info('health indicator plotting start')
        
            cluster_distances = np.concatenate((train_distances,test_distances))

            colors = ["r" if status > threshold_list[3] else "g" for status in cluster_distances]

            plt.figure(figsize=(10, 8))
            plt.style.use('ggplot')
            plt.scatter(range(len(cluster_distances)),cluster_distances,c = colors,s=4,alpha=0.7,label='Min Distances')
            plt.plot(cluster_distances,color='gray',alpha=0.3, linestyle='-',linewidth=0.5)

            plt.hlines(y=threshold_list[0], xmin=0, xmax=len(cluster_distances),alpha=0.5,linewidth=1, colors='lightseagreen',linestyles='dashed',label='90% Percentile Threshold')
            plt.hlines(y=threshold_list[1], xmin=0, xmax=len(cluster_distances), alpha=0.5,linewidth=1,colors='darkslategray', linestyles='dashed', label='95% Percentile Threshold')
            plt.hlines(y=threshold_list[2], xmin=0, xmax=len(cluster_distances),alpha=0.5, linewidth=1,colors='mediumblue', linestyles='dashed', label='97% Percentile Threshold')
            plt.hlines(y=threshold_list[3], xmin=0, xmax=len(cluster_distances), alpha=0.5,linewidth=1,colors='blueviolet', linestyles='dashed', label='99% Percentile Threshold')

            plt.xticks(np.arange(0, len(date), 100), date[::100], rotation=90, size=8)
            plt.title(f'Euclidean Distances for Each Data Point \n for {self.test_name} bearing {self.bearing_number}')
            plt.xlabel('Timestamp')
            plt.ylabel('Euclidean Distances')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.save_plot)
            plt.show()

            logging.info('health indicator plotting end')
        
        except Exception as e:
            raise CustomException(e,sys)





    