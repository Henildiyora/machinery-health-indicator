from src.exception import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_path:str = os.path.join('artifacts','data.csv')

class Dataingestion:

    def __init__(self):
        self.dataIngestionConfig = DataIngestionConfig()


    def initiate_data_ingestion(self,train_date):
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

            time_df = pd.read_csv('artifacts/timedomaian.csv')
            time_df = time_df.drop(columns='Date')
            
            freq_df = pd.read_csv('artifacts/frequencydomain.csv')

            df = pd.concat([time_df,freq_df],axis=1)

            date = df['Date'].values

            #split the date into training and testing sets
            df_train = df[df['Date'] <= train_date]
            df_test = df[df['Date'] > train_date]

            df.to_csv(self.dataIngestionConfig.data_path,index=False,header=True)

            logging.info('Data ingestion finish')

            return df_train,df_test,date

        except Exception as e:
            raise CustomException(e,sys)


# if __name__ == "__main__":
    
#     obj = Dataingestion()

#     df_train,df_test,date = obj.initiate_data_ingestion(train_date='2004.02.12.11.12.39')
#     print(f'df train = \n{df_train}')
#     print(f'df_test = \n{df_test}')
#     print(f'date = \n {date}')
    