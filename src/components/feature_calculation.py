from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from src.utils import read_raw_datafiles, calculate_fft , save_csv


class FeatureCalculation:
    '''
    calculate the timedomain and frequencydomain features 
    '''
    def __init__(self,test_name,bearing_number):
        self.test_name : str = test_name
        self.bearing_number : int= bearing_number
        self.timedf_path = os.path.join('artifacts',f'{test_name}_bearing_{bearing_number}_timedomaian.csv')
        self.freqdf_path = os.path.join('artifacts',f'{test_name}_bearing_{bearing_number}_frequencydomain.csv')

    def calculate_time_domain_features(self,file_path,bearing_number):
        '''
        function calculate the time domain features from raw data 
        Args:
            file_path = path of raw_data 
            bearing_number = for which bearing wants to calculate timedomain features 

        Reaturn:
            data frame that include calculated time domain features
        '''
        logging.info('time domain feature calculation start')

        try:

            # extract the raw data
            values_array,date = read_raw_datafiles(file_path,bearing_number)

            time_domain_list = []

            for i in range(len(values_array)):

                mean_ = np.mean(abs(values_array[i]))
                standard_deviation = np.std(values_array[i])
                RMS = np.sqrt(np.mean(np.square(values_array[i])))
                peak_value = np.max(values_array[i])
                skewness = skew(values_array[i])
                kurt = kurtosis(values_array[i])
                crest_factor = peak_value / RMS
              

                timedomain_dict = {
                    'mean' : mean_,
                    'std'  : standard_deviation,
                    'root_mean_squre' : RMS,
                    'max' : peak_value,
                    'skewness' : skewness,
                    'kurtosis' : kurt,
                    'crest_factor' : crest_factor,
                }

                time_domain_list.append(timedomain_dict)


            # Convert the list of dictionaries to a DataFrame
            time_df = pd.DataFrame(time_domain_list) 

            # Add dates line by line to DataFrame
            for index, date in enumerate(date):
                time_df.loc[index, 'Date'] = date

            save_csv(time_df,file_path=self.timedf_path)

            logging.info('time domain feature calculation finish')

            return time_df  

        except Exception as e:
            raise CustomException(e,sys)
        
    def cal_frequency_domain_features(self,file_path,bearing_number):
        '''
        function calculate the frequency domain features from raw data 
        Args:
            file_path = path of raw_data 
            bearing_number = for which bearing wants to calculate frequency domain features 

        Reaturn:
            data frame that include calculated frequencydomain features
        '''

        values_array,date = read_raw_datafiles(file_path,bearing_number)

        freq_domain_list = []

        try:
            
            logging.info('frequency domain feature calculation start')

            for i in range(len(values_array)):

                fft_data = calculate_fft(values_array[i])

                mean_ = np.mean(fft_data)
                variance_of_mean_frequency = np.mean(np.square(fft_data - mean_))
                skewness_power_spectrum = skew(fft_data)
                kurtosis_power_spectrum = kurtosis(fft_data)
                frequency_rms = np.sqrt(np.mean(np.square(fft_data)))
                root_variance = np.sqrt(np.var(fft_data))

                freq_domain_dict = {
                    'freq_mean' : mean_,
                    'freq_variance_of_mean_frequency' : variance_of_mean_frequency,
                    'freq_skewness' : skewness_power_spectrum,
                    'freq_kurtosis' : kurtosis_power_spectrum,
                    'freq_rms' : frequency_rms,
                    'root_variance' : root_variance
                }

                freq_domain_list.append(freq_domain_dict)

            freq_df = pd.DataFrame(freq_domain_list)

            # Add dates line by line to DataFrame
            for index, date in enumerate(date):
                freq_df.loc[index, 'Date'] = date

            save_csv(freq_df,file_path=self.freqdf_path)

            logging.info('frequency domain feature calculation finish')

            return freq_df
        
        except Exception as e:
            raise CustomException(e,sys)
    




        



        


