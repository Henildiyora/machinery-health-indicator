from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
import sys
import os
import pickle
import json

def read_raw_datafiles(file_path:str,bearing_number:int):
    '''
    extract the raw data from text files 
    Args:
        file_path = text file path
        bearing_number = for which bearing wants to calculate timedomain features 

    Return:
        extracted values in np.array form and file names for date and time 
    '''
    try:

        logging.info('raw data extraction start')

        # For extracting actual bearing number
        bearing_number = bearing_number - 1 
        value_list = []
        date = []

        # List out all files
        files = os.listdir(file_path)

        for file in sorted(files):

            read_file = np.loadtxt(os.path.join(file_path,file))
            read_file = read_file[:,bearing_number]
            read_file = read_file - np.mean(read_file)
            value_list.append(read_file)
            date.append(file)

        logging.info('raw data extraction finish')

        return np.array(value_list),date
    
    except Exception as e:
        raise CustomException(e,sys)


# read_file = read_raw_datafiles('raw_data/1st_test/1st_test',2)
# print(read_file[0])
    
def calculate_fft(raw_data,fs=20000):
    '''
    function calculate the FFT of raw signal 
    Args:
        raw_data : raw signal 
        fs : sample rate 
    
    Return:
        x : calculated fft of raw data 
    '''
    try:

        x = np.fft.fft(raw_data)
        x = abs(x)

    except Exception as e:
        raise CustomException(e,sys)


    return x

def save_object(file_path,obj):
    '''
    function save any object into pickle file
    Args:
        file_path = file save location path
        obj = which object want to save (scaler,model)

    '''
    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    '''
    function load the saved pickle objects
    '''
    try:
        with open(file_path,'rb') as file_obj:
            file = pickle.load(file_obj)

        return file
    
    except Exception as e:
        raise CustomException(e,sys)
    
def save_json(file_path,obj):
    '''
    function save any object into json file
    '''
    try:
    
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'w') as python_obj:
            json.dump(obj,python_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def load_json(file_path):

    try:
        
        with open(file_path,'r') as python_obj:
            file = json.load(python_obj)

        return file
    
    except Exception as e:
        raise CustomException(e,sys)

def save_csv(df,file_path):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        df.to_csv(file_path,index=False,header=True)

    except Exception as e:
        raise CustomException(e,sys)



    










