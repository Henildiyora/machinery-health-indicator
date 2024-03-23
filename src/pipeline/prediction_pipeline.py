import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import pairwise_distances
from src.utils import load_object,save_json
import sys,os


class NewDataPointDistances:

    def __init__(self,test_name,bearing_number) -> None:
        self.test_name = test_name
        self.bearing_number = bearing_number
        self.save_distance = os.path.join('new_distances',f'{test_name}_bearing_{bearing_number}_newdistances.json')


    def new_datapoint_distance(self,saved_model_path,new_data,saved_scaler_path = None,):

        try:

            logging.info('new data valus distances calculation start')

            model = load_object(saved_model_path)

            if saved_scaler_path is not None:
                scaler = load_object(saved_scaler_path)

                new_data = scaler.transform(new_data)

            cluster_centers = model.cluster_centers_

            distances = pairwise_distances(new_data,cluster_centers)

            print(distances)

            distances_object = {'distances':distances.tolist()}

            save_json(self.save_distance,distances_object)

            logging.info('new data valus distances calculation end')

        except Exception as e:
            raise CustomException(e,sys)


new_data = np.random.rand(100,13)
test_name = '1st_test'
bearing_number = 1
obj = NewDataPointDistances(test_name=test_name,bearing_number=bearing_number)
obj.new_datapoint_distance(saved_model_path='save_models/1st_test_bearing_2_model.pkl',
                    saved_scaler_path='save_scaler/1st_test_bearing_2_scaler.pkl',
                    new_data=new_data)



        

        

        




