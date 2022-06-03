# -*- coding: utf-8 -*-
"""
Created on Thr May 26 2022

@author: David Francisco Dávila Ortega


This is a web app to classify the inpatient length of Stay in hospitals based on\
          several features that you can see in the sidebar. Please adjust the\
          value of each feature. After that, click on the Predict button at the bottom to\
          see the prediction of the classifier.

"""

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]
    
model = load_model('./catboost_71')

st.title('Length of Stay Classifier Web App')

st.write('Esta es una aplicación web para clasificar la duración de la estadía\n\
de los pacientes en los hospitales según las características que se pueden ver en la barra lateral.\n\
Selecciona un valor para cada opción y, después haz clic en el botón `Predecir` de la parte inferior\n\
para ver la predicción del clasificador.')

# ['year', 'hospital', 'sex', 'admission_types', 'service_code',
#        'section_code', 'reason_of_discharge', 'diagnostic_code1',
#        'therapy_code1', 'Age', 'Length_of_Stay_Bin_qcut'],
#       dtype='object')

year = st.sidebar.slider(label = 'Año', min_value = 2005,
                          max_value = 2022 ,
                          value = 2010,
                          step = 1)

hospital = st.sidebar.text_input('COMPLEJO ASISTENCIAL DE LEON')

# .selectbox('Selecciona un hospital', ['COMPLEJO ASISTENCIAL DE AVILA', 'COMPLEJO ASISTENCIAL DE BURGOS',
#        'HOSPITAL EL BIERZO', 'COMPLEJO ASISTENCIAL DE LEON',
#        'COMPLEJO ASISTENCIAL DE PALENCIA',
#        'COMPLEJO ASISTENCIAL DE SALAMANCA',
#        'COMPLEJO ASITENCIAL DE SEGOVIA', 'COMPLEJO ASISTENCIAL DE SORIA',
#        'HOSPITAL CLINICO UNIVERSITARIO DE VALLADOLID',
#        'COMPLEJO ASISTENCIAL DE ZAMORA',
#        'HOSPITAL UNIVERSITARIO DEL RIO HORTEGA'])
                          
sex = st.sidebar.radio('Selecciona un género', ['MUJER', 'HOMBRE'])                   

admission_types = st.sidebar.selectbox('Selecciona el tipo de admisión', ['INGRESO URGENTE', 'INGRESO PROGRAMADO'])


service_code = st.sidebar.text_input('PSQ')
   
   # PSQ
section_code = st.sidebar.text_input('URSM')

# ['PSQ', 'HPSQ', 'PSI', 'PSQH', 'URSM', 'PSQP', 'PSQ2', 'REHP', 'L',
#        'UCPS', 'PSRH', 'PSCH', 'HPDS', 'PSIJ', 'URAP', 'UCAP', 'UCSM',
#        '-', 'PSQM', 'RPSS', 'URH', 'UDCP', 'UPD', 'URS', 'ULE']

reason_of_discharge = st.sidebar.text_input('EXITUS')

# ['DOMICILIO', 'ALTA VOLUNTARIA', 'EXITUS',
#        'TRASLADO A OTRO HOSPITAL', 'NO ESPECIFICADO',
#        'TRASLADO A CENTROS DE MEDIA Y LARGA ESTANCIA', 'OTROS',
#        'ALTA SIN NOTIFICACIÓN DOCUMENTADA']

diagnostic_code1 = st.sidebar.text_input('296.81')

# 295.64', '296.22', '291.81', '295.62', '300.00', '295.60',
#        '293.0', '312.9', '303.91', '250.11', '307.51', '296.30', '296.00',
#        '296.60', '309.9', '296.7', '305.00', '308.9', '300.9', '295.94',
#        '300.82', '309.0', '295.30', '296.81', '301.83', '294.9', '295.74',
#        '319', '305.90', '309.4', '294.8', '301.0', '303.01', '799.2',
#        '307.1', '297.1', '300.19', '564.89', '295.33', '295.63', '296.44',
#        '824.2', '296.33', '300.15', '292.12', '296.50', '305.21',
#        '296.42', '969.4', '295.92', '300.7', '295.32', '296.32', '295.

therapy_code1 = st.sidebar.text_input('93.96')
                          
       #                    88.91', '96.59', '93.96', '93.54', '93.56', '44.13', '93.59',
       # '89.14', '88.71', '88.75', '57.94', '88.73', '88.79', '93.90',
       # '31.42', '87.41', '93.99', '86.59', '79.15', '16.21', '99.04',
       # '88.93', '81.91', '99.29', '87.62', '87.64', '21.01', '38.93',
       # '88.98', '89.54', '88.77', '99.19', '03.31', '94.11', '94.13',
       # '45.13', '54.75', '95.11', '54.11', '88.38', '99.18', '87.44',
       # '90.55', '90.59', '91.35', '89.65', '99.26', '96.6', '39.95',
       # '96.71', '89.06', '91.33', '96.33', '96.35', '94.39', '94.38',
       # '89.29', '88.78', '91.32', '91.55', '44.42', '99.21', '90.49',
       # '94.63', '94.61', '89.51', '90.53', '94.62', '94.66', '94.65'

Age = st.sidebar.number_input('Elige la edad', 0, 102)

# slider(label = 'Edad', min_value = 0,
#                           max_value = 102,
#                           value = 35,
#                           step = 1)

# Display interactive widgets
# st.button('Click me')
# st.checkbox('I agree')
# st.radio('Pick one', ['cats', 'dogs'])
# st.selectbox('Pick one', ['cats', 'dogs'])
# st.multiselect('Buy', ['milk', 'apples', 'potatoes'])
# st.slider('Pick a number', 0, 100)
# st.select_slider('Pick a size', ['S', 'M', 'L'])
# st.text_input('First name')
# st.number_input('Pick a number', 0, 10)
# st.text_area('Text to translate')
# st.date_input('Your birthday')
# st.time_input('Meeting time')
# st.file_uploader('Upload a CSV')
# st.download_button('Download file', data)
# st.camera_input("Take a picture")
# st.color_picker('Pick a color')


features = {'year': year, 'hospital': hospital, 'sex': sex, 'admission_types': admission_types, 
			'service_code': service_code, 'section_code': section_code, 'reason_of_discharge': reason_of_discharge, 
			'diagnostic_code1': diagnostic_code1, 'therapy_code1': therapy_code1, 'Age': Age}
			# 'Length_of_Stay_Bin_qcut': Length_of_Stay_Bin_qcut}

features_df  = pd.DataFrame([features])

# Create dummy columns for categorical variables
# prefix_cols = ['ADM', 'SEX', 'ROD', 'DC1', 'AGE', 'TC1', 'LOS', 'HOS', 'SRVC', 'SCTC']
# dummy_cols = ['admission_types', 'sex', 'reason_of_discharge', 
#               'diagnostic_code1', 'Age', 'therapy_code1', 'Length_of_Stay_Bin_qcut'
#               , 'hospital', 'service_code', 'section_code']

# df2 = pd.get_dummies(features_df, prefix=prefix_cols, columns=dummy_cols)

st.table(features_df)  

if st.button('Predecir'):
    
    prediction = predict_quality(model, features_df)

    st.write('Basado en los parámetros seleccionados, \
        se sugiere que el/la paciente tendría una:\n'+ str(prediction))

# Run with: streamlit run app.py

# y_pred=clf.predict(X_test)