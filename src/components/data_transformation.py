from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils import save_obj
from src.config.configuration import PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH,FEATURE_ENG_OBJ_PATH

class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        logging.info("Feature Engneering Started")

    def calculate_distance(self,df,lat1, lon1, lat2, lon2):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12742 * np.arcsin(np.sqrt(a))
        
    def transform_data(self,df):
        try:
            logging.info("Calclating distance using latitude and longitude")
            self.calculate_distance(df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')

            df.drop(['ID','Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude',
                        'Order_Date','Time_Orderd','Time_Order_picked'],axis=1,inplace=True)
              
            logging.info(f'Unnecessary columns droped')

            return df

        except Exception as e:
            logging.info("Error ocured in Data Transformation process")
            raise CustomException(e, sys) from e 
        

    def fit(self,X,y=None):
        return self
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e



@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path=TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path=TRANSFORMED_TEST_FILE_PATH
    feature_eng_obj_path=FEATURE_ENG_OBJ_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation started')

            categorical_column=['Type_of_order','Type_of_vehicle','Road_traffic_density','Festival','City','Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']
            
            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler(with_mean=False))
            ])
            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
            ])

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_column),
                ('categorical_pipeline',categorical_pipeline,categorical_column),
                ])

            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("error occured in data transformation process")
            raise CustomException(e,sys)

    def get_feature_engineering_object(self):
        try:
            feature_engineering = Pipeline(steps = [("feature_engineering_step",Feature_Engineering())]) 
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e


    def initaite_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()

            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)
            logging.info(f"Feature engineering object has been applied  on training and testing dataframe")


            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")
            logging.info(f"Saving csv to train_data and test_data.csv")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'

            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=target_column_name,axis=1)
            y_test=test_df[target_column_name]

            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)
            logging.info("Applying preprocessing object on training and testing datasets.")

            logging.info(" Data Transformation completed")

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            logging.info("converting train_arr and test_arr to dataframe")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {df_train.columns}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path,index=False,header=True)


            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")


            save_obj(
                file_path=self.data_transformation_config.feature_eng_obj_path,
                obj=fe_obj)
            logging.info("Feature eng file saved")

            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)



        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
