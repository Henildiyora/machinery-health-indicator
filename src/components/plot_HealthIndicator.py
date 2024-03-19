# from sklearn.preprocessing import StandardScaler,MinMaxScaler

# from src.components.data_ingestion import Dataingestion
# from src.components.model_trainer import TrainModel
# from src.components.cal_distancesOfTrainDataPoint import CalculateDistancesTrain
# from src.components.cal_distancesOfNewDataPoint import CalculateDistancesNewDataPoint

# import matplotlib.pyplot as plt
# import numpy as np


# if __name__ == "__main__":

#     dataingestion = Dataingestion()
#     df_train,df_test,date = dataingestion.initiate_data_ingestion(train_date='2004.02.15.13.12.39')

#     trainmodel = TrainModel()
#     scaler = MinMaxScaler()
#     df_label,cluster_centers = trainmodel.train_model(df_train=df_train,df_test=df_test,n_cluster=5,scaler=scaler)

#     cal_train_distance = CalculateDistancesTrain()
#     metric = 'euclidean'
#     min_distances,max_distances,percentile_distances,train_minimum_distances = cal_train_distance.calculate_distances(df_train=df_train,cluster_centers=cluster_centers,metric=metric,scaler=True)
    
#     cal_new_datapoint_distance = CalculateDistancesNewDataPoint()
#     new_data_min_distance = cal_new_datapoint_distance.calculate_distance_new_datapoint(new_data=df_test,cluster_centers=cluster_centers,metric=metric,scaler=True)

#     print(f'train_min_distances = {min_distances}')
#     print(f'max_distances = {max_distances}')
#     print(f'percentile_distances = {percentile_distances}')
#     print(f'train_minimum_distances = {train_minimum_distances}')

#     min_mean = np.mean(min_distances)
#     percentile_mean = np.mean(percentile_distances)
#     max_mean = np.mean(max_distances)

#     print(f'min_mean = {min_mean}')
#     print(f'percentile_mean = {percentile_mean}')
#     print(f'max mean = {max_mean}')

#     distance = np.concatenate((train_minimum_distances,new_data_min_distance))



#     def plot_distances(min_distances,min_threshold, max_threshold,percentile_threshold,timestamp,new_pred=None):
#         """Plot the max distances for each data point along with the mean distance on a dashed red line.

#         Args:
#             max_distances (list): Max distance for each data point from all cluster centroids
#             mean_distance_train (float): Mean distance calculated during training
#             timestamp (list): List of timestamps corresponding to data points
#         """


#         colors = ["r" if status > percentile_threshold else "g" for status in min_distances]


#         plt.figure(figsize=(10, 8))
#         plt.style.use('ggplot')
#         plt.scatter(range(len(min_distances)),min_distances,c = colors,s=4,alpha=0.7,label='Min Distances')
#         plt.plot(min_distances,color='gray',alpha=0.3, linestyle='-',linewidth=0.5)
#         # plt.plot(new_pred, label='one class svm pridction', color='r', alpha=0.5, linestyle='-', marker='o', markersize=2, linewidth=0.5)
#         if new_pred:
#             plt.scatter(np.arange(len(new_pred)), new_pred,color='blue',s=5, label='one class svm anomalies')

#         # plt.hlines(y=max_threshold, xmin=0, xmax=len(min_distances), colors='red', linestyles='dashed', label='Max Threshold')
#         # plt.hlines(y=max_threshold, xmin=0, xmax=len(min_distances), colors='red', linestyles='dashed', label='max Threshold')
#         plt.hlines(y=min_threshold, xmin=0, xmax=len(min_distances), colors='green', linestyles='dashed', label='min Threshold')
#         plt.hlines(y=percentile_threshold, xmin=0, xmax=len(min_distances), colors='blue', linestyles='dashed', label='Percentile Threshold')


#         plt.xticks(np.arange(0, len(timestamp), 100), timestamp[::100], rotation=90, size=8)
#         plt.title(f'Euclidean Distances for Each Data Point')
#         plt.xlabel('Timestamp')
#         plt.ylabel('Euclidean Distances')
#         # plt.ylim(0,5)
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     plot_distances(min_distances=distance,
#                    min_threshold=min_mean,
#                    max_threshold=max_mean,
#                    percentile_threshold=percentile_mean,
#                    timestamp=date)
    

    



