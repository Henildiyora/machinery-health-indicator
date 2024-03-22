from src.exception import CustomException
from src.logger import logging
from src.utils import load_json
from src.components.data_ingestion import Dataingestion
from src.components.model_trainer import ModelTraining
from src.components.cal_distancesOfTrainDataPoint import CalculateDistancesTrain
from src.components.cal_distancesOfNewDataPoint import CalculateDistancesNewDataPoint
from src.components.feature_calculation import FeatureCalculation
from src.components.plot_HealthIndicator import PlotHealthIndicator

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":

    test_name = '1st_test'
    bearing_number = 4

    feature_calculation = FeatureCalculation(test_name,bearing_number)
    feature_calculation.calculate_time_domain_features(file_path=f'raw_data/{test_name}/{test_name}',bearing_number=bearing_number)
    feature_calculation.cal_frequency_domain_features(file_path=f'raw_data/{test_name}/{test_name}',bearing_number=bearing_number)


    data_ingestion = Dataingestion(test_name,bearing_number)
    df_train,df_test,date = data_ingestion.initiate_data_ingestion(train_date='2003.11.08.12.01.44',
                                                                   timedf_path=f'artifacts/{test_name}_bearing_{bearing_number}_timedomaian.csv',
                                                                   freqdf_path=f'artifacts/{test_name}_bearing_{bearing_number}_frequencydomain.csv')

    model_training = ModelTraining(test_name,bearing_number)
    scaler = MinMaxScaler()
    mertic = 'euclidean'
    df_label,cluster_centers = model_training.model_training(df_train=df_train,df_test=df_test,n_cluster=6,scaler=scaler)

    train_distances = CalculateDistancesTrain(test_name,bearing_number)
    train_distances.calculate_distances(df_train=df_train,cluster_centers=cluster_centers,metric=mertic,scaler=True)

    test_distances = CalculateDistancesNewDataPoint(test_name,bearing_number)
    test_distances.calculate_distance_new_datapoint(new_data=df_test,cluster_centers=cluster_centers,metric=mertic,scaler=True)

    train_distance_detail = load_json(file_path=f'distances/{test_name}_bearing_{bearing_number}_trainDistances.json')
    test_distance_detail = load_json(file_path=f'distances/{test_name}_bearing_{bearing_number}_testDistances.json')

    # print('train_distance_detail = ',train_distance_detail)
    # print('test_distance_detail = ',test_distance_detail)

    train_distances = train_distance_detail['train_distances']
    training_dates = train_distance_detail['training_dates']
    threshold_list = train_distance_detail['threshold']
    test_distances = test_distance_detail['test_distances']
    testing_dates = test_distance_detail['testing_dates']

    date = training_dates + testing_dates

    health_indicator = PlotHealthIndicator(test_name,bearing_number)
    health_indicator.plot_health_indicator(train_distances=train_distances,test_distances=test_distances,
                                           threshold_list=threshold_list,date=date)


