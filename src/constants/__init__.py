import os 
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()


# Current Directiory
ROOT_DIR_KEY = os.getcwd()

# Data Storage File path 
DATA_DIR = "Data"
DATA_DIR_KEY = 'Delivery.csv'
ARTIFACT_DIR_KEY = "Artifact"

#Data Ingestion 
DATA_INGESTION_KEY = 'data_ingestion'
DATA_INGESTION_RAW_DATA_DIR_KEY= 'raw_data_dir'
RAW_DATA_DIR_KEY = 'raw.csv'

DATA_INGESTION_INGESTED_DIR_NAME_KEY= 'ingested_dir'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv' 

# Data validation
DATA_VALIDATION_KEY = 'data_validation'
DATA_VALIDATION_VALIDATED_DIR_KEY = 'validated_dir'
VALIDATED_TRAIN_DIR_KEY = 'validated_train.csv'
VALIDATED_TEST_DIR_KEY = 'validated_test.csv'


#Data Transformation
DATA_TRANSFORMATION_ARTIFACT = 'data_transformation'
DATA_PREPROCESSED_DIR='preprocessed'
DATA_TRANSFORMATION_PREPROCESSING_OBJ = 'preprocessor.pkl'

DATA_TRANSFORMED_DIR = 'transformed_data'
TRANSFORMED_TRAIN_DIR_KEY = 'train.csv'
TRANSFORMED_TEST_DIR_KEY = 'test.csv'

#Model Training
MODEL_TRAINER_KEY = 'model_trainer'
MODEL_OBJECT = 'model.pkl'