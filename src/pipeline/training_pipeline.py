from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

class Train:
    def __init__(self):
        self.c = 0

    def main(self):
        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
        model_trainer=ModelTrainer()
        model_trainer.initate_model_training(train_arr,test_arr)

