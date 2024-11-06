from flask import Flask, request,render_template,redirect,url_for
import numpy as np
import pandas as pd
from src.components.data_transformation import DataTransformationConfig
from src.config.configuration import *
from src.pipeline.training_pipeline import Train
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.logger import logging
import os


model= MODEL_FILE_PATH
feature_eng= FEATURE_ENG_OBJ_PATH
transformer= PREPROCESSING_OBJ_PATH

app= Flask(__name__,template_folder='templates')
app.static_folder = 'static'
app.static_url_path = '/static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_time():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Delivery_person_Age = int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition =  int(request.form.get('Vehicle_condition')),
            multiple_deliveries = int(request.form.get('multiple_deliveries')),
            distance = float(request.form.get('distance')),
            Type_of_order = request.form.get('Type_of_order'),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            Festival = request.form.get('Festival'),
            City = request.form.get('City')
        )
        df= data.get_data_as_dataframe()
        pipe= PredictPipeline()
        p=pipe.predict(df)
        
        ans= int(p[0])
        
        return render_template('result.html',result=ans)
    
if __name__=='__main__':
    app.run(host='0.0.0.0')
    # app.run(host='0.0.0.0',debug=True, port ='8888')
    
