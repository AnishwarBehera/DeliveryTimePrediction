from src.constants import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionconfig:
    train_data_path:str=TRAIN_FILE_PATH
    test_data_path:str=TEST_FILE_PATH
    raw_data_path:str= RAW_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Initiate Data Ingestion config")
        logging.info("==============================")
        
        try:
            df=pd.read_csv(DATASET_PATH)
            logging.info('Data reading has been completed')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)  
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)   

            logging.info("train test split started")
            train_set,test_Set = train_test_split(df,test_size=0.20)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
           
            logging.info("train data saved")
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)
            test_Set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info("test data saved")

            logging.info("Data ingestion complete")


            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
  
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)

 

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data_path,test_data_path)
#     model_trainer = ModelTrainer()
#     print(model_trainer.initate_model_training(train_arr,test_arr))
