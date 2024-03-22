from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from src.utils import save_csv

class Dataingestion:

    '''
    devide the data into traiining and testing sets 
    '''

    def __init__(self,test_name,bearing_number):
        self.test_name : str= test_name
        self.bearing_number : int = bearing_number
        self.dataPath = os.path.join('artifacts',f'{test_name}_bearing_{bearing_number}_data.csv')


    def initiate_data_ingestion(self,train_date,timedf_path,freqdf_path):
        '''
        Function to divide the data into training and testing sets.

        Args:
            train_date: Data up to which the model will be trained.
            train_date Formate = 'year.month.date.hour.minute.second'

        Returns:
            df_train: Training set.
            df_test: Testing set.
            date: Dates of the data.
        '''
        logging.info('data ingesion start')

        try:

            # read the both csv file and drop date column in one csv file 
            time_df = pd.read_csv(timedf_path)
            time_df = time_df.drop(columns='Date')
            
            freq_df = pd.read_csv(freqdf_path)

            df = pd.concat([time_df,freq_df],axis=1)

            date = df['Date'].values

            #split the date into training and testing sets
            df_train = df[df['Date'] <= train_date]
            df_test = df[df['Date'] > train_date]

            save_csv(df,file_path=self.dataPath)

            logging.info('Data ingestion finish')

            return df_train,df_test,date

        except Exception as e:
            raise CustomException(e,sys)


