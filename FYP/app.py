# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 13:04:28 2022

@author: USER
"""


import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

st.set_page_config(page_title="MDPS", page_icon="card-list.png", layout="centered", initial_sidebar_state="auto", menu_items=None)

@st.cache_data
def convert_df(Rd):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return Rd.to_csv().encode('utf-8')

def download(content, fn, K):
    st.download_button(
        label="Download Report for each Patient",
        data=content,
        file_name=fn,
        mime='text/csv',
        key = K
    )

# loading the saved models

diabetes_model = pickle.load(open('/models/diabetes_model_u10.sav', 'rb'))

heart_model = pickle.load(open('/models/heart_model.sav', 'rb'))

liver_model = pickle.load(open('/models/liver_model_u.sav', 'rb'))

kidney_model = pickle.load(open('/models/kidney_model_u13.sav', 'rb'))

diabKeys = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
heartKeys = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
liverKeys = ['Total_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','AlbuminL']
kidneyKeys = ['AlbuminK','Blood_Urea','Serum_Creatinine','Sodium','Potassium']

dict_of_diseases= dict(diab= diabKeys, heart= heartKeys, liver= liverKeys, kidney= kidneyKeys)

#Function to check for validating inputs
def valid_input(inputs):
    for inp in inputs:
        try:
            float(inp)
            # print(inp)
        except:
            return False
    return True

#Function to clear form on button click
def clear_form(disease):
	for key in dict_of_diseases[disease]:
		 st.session_state[key]=""


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                            ['Details given by User',
                            'Details fetched from CSV File'
                            # 'Visualiztion'
                            ],
                            icons=['person','bi-file-earmark-arrow-down'],
                                #    ,'bi-graph-up'],
                            default_index=0)
    

if (selected == 'Details given by User'):
    #Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Diabetes Prediction", "Heart Disease Prediction", "Liver Disease Prediction", "Kidney Disease Prediction"])   
    
    # Diabetes Prediction Page
    # if (selected == 'Diabetes Prediction'):
    with tab1:
        # page title
        st.title('Diabetes Prediction')
        
        submitted = False
        clear = False
        
        with st.form(key="diabetesForm"):
            # getting the input data from the user
            col1, col2, col3 = st.columns(3)
            
            with col1:
                Pregnancies = st.text_input('Number of Pregnancies', key='Pregnancies')
                
            with col2:
                Glucose = st.text_input('Glucose level',key='Glucose')
            
            with col3:
                BloodPressure = st.text_input('Diastolic Blood Pressure (mm Hg)',key='BloodPressure')
            
            with col1:
                SkinThickness = st.text_input('Skin Thickness (mm)',key='SkinThickness')
            
            with col2:
                Insulin = st.text_input('Insulin (mu U/ml)',key='Insulin')
            
            with col3:
                bmi = st.text_input('BMI',key='BMI')
            
            with col1:
                DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value',key='DiabetesPedigreeFunction')
            
            with col2:
                Age = st.text_input('Age',key='Age')
            
            
            # code for Prediction
            diab_diagnosis = ''
            
            inputData = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, bmi, DiabetesPedigreeFunction, Age]
            # print(inputData)

            # creating buttons for Prediction and Clear
            c1, c2, c3 = st.columns([3, 1, 1],gap="large")
            
            with c1:
                submitted = st.form_submit_button(label="Diabetes Test Result")
            with c2:
                pass
            with c3:
                clear = st.form_submit_button(label="Clear",on_click=clear_form,args=("diab", ))
            
            
            
        if submitted:
            if(valid_input(inputData)): 
                # print(inputData)  
                diab_prediction = diabetes_model.predict([[float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(bmi), float(DiabetesPedigreeFunction), float(Age)]])
                

                if (diab_prediction[0] == 0):
                    diab_diagnosis = 'The person is not diabetic'
                    st.success(diab_diagnosis)
                else:
                    diab_diagnosis = 'The person is diabetic'
                    st.error(diab_diagnosis)
                
            else:
                st.error("All fields are required and must have appropriate values")
        

    # Heart Disease Prediction Page: Used RF model 94.634% accuracy
    # =============================================================================
    # if (selected == 'Heart Disease Prediction'):
    with tab2:
        #page title
        st.title('Heart Disease Prediction')
        
        validSelect = True
        submitted = False
        clear = False
        
        with st.form(key="heartForm"):
                
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.text_input('Age',key='age')
                
            with col2:
                sex = st.selectbox('Gender',('','Male', 'Female'),key='sex')
                
            with col3:
                cp = st.selectbox('Chest Pain Level',('','Absent', 'Faint', 'Moderate', 'Severe'),key='cp')
                
            with col1:
                trestbps = st.text_input('Resting Blood Pressure (mm Hg)',key='trestbps')
                
            with col2:
                chol = st.text_input('Serum Cholestoral (mg/dL)',key='chol')
                
            with col3:
                fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',('', 'Yes', 'No'),key='fbs')
                
            with col1:
                restecg = st.selectbox('Resting Electrocardiographic results',('', '0', '1', '2'),key='restecg')
                
            with col2:
                thalach = st.text_input('Maximum Heart Rate achieved',key='thalach')
                
            with col3:
                exang = st.selectbox('Exercise Induced Angina',('', 'Yes', 'No'),key='exang')
                
            with col1:
                oldpeak = st.text_input('ST depression induced by exercise',key='oldpeak')
                
            with col2:
                slope = st.text_input('Slope of peak exercise ST segment',key='slope')
                
            with col3:
                ca = st.selectbox('Major vessels colored by flourosopy',('' ,'0', '1', '2', '3'),key='ca')
                
            with col1:
                thal = st.selectbox('Thalassemia', ('', 'Not examined', 'Normal', 'Fixed Defect', 'Reversable Defect'),key='thal')
            
            if(sex == 'Male'):
                sex = '1'
            elif(sex == 'Female'):
                sex = '0'
            else:
                validSelect = False
                
            if(cp=='Absent'):
                cp = '0'
            elif(cp=='Faint'):
                cp = '1'
            elif(cp=='Moderate'):
                cp = '2'
            elif(cp=='Severe'):
                cp = '3'
            else:
                validSelect = False
        
            if(fbs=='Yes'):
                fbs = '1'
            elif(fbs=='No'):
                fbs = '0'
            else:
                validSelect = False
                
            if(exang=='Yes'):
                exang = '1'
            elif(exang=='No'):
                exang = '0'
            else:
                validSelect = False
                
            if(thal=='Not examined'):
                thal = '0'
            elif(thal=='Normal'):
                thal = '1'
            elif(thal=='Fixed Defect'):
                thal = '2'
            elif(thal=='Reversable Defect'):
                thal = '3'
            else:
                validSelect = False
                
            # code for Prediction
            heart_diagnosis = ''
            
            inputData = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            
            # creating buttons for Prediction and Clear
            c1, c2, c3 = st.columns([3, 1, 1],gap="large")
            
            with c1:
                submitted = st.form_submit_button(label="Heart Disease Test Result")
            with c2:
                pass
            with c3:
                clear = st.form_submit_button(label="Clear",on_click=clear_form,args=("heart", ))
            
            
            if submitted:
                if(valid_input(inputData) and validSelect==True):   
                    heart_prediction = heart_model.predict([[age,sex,cp,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
                        
                    if (heart_prediction[0] == 0):
                        heart_diagnosis = 'The person does not have heart disease'
                        st.success(heart_diagnosis)
                    else:
                        heart_diagnosis = 'The person has some heart disease'
                        st.error(heart_diagnosis)
                else:
                    st.error("All fields are required and must have appropriate values")
            
            if clear:
                validSelect = True

    # Liver Disease Prediction Page --- used rf model : 95.348 accuracy

    # if (selected == 'Liver Disease Prediction'):
    with tab3:
        #page title
        st.title('Liver Disease Prediction')
        
        submitted = False
        clear = False
        
        with st.form(key="liverForm"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                Total_Bilirubin = st.text_input('Total Bilirubin (mg/dL)',key='Total_Bilirubin')
            
            with col2:
                Alkaline_Phosphotase = st.text_input('Alkaline Phosphotase (IU/L)',key='Alkaline_Phosphotase')
            
            with col3:
                Alamine_Aminotransferase = st.text_input('Alamine Aminotransferase (IU/L)',key='Alamine_Aminotransferase')
            
            with col1:
                Aspartate_Aminotransferase = st.text_input('Aspartate Aminotransferase (IU/L)',key='Aspartate_Aminotransferase')
            
            with col2:
                Albumin = st.text_input('Albumin (g/dL)',key='AlbuminL')
                    
            # code for Prediction
            liver_diagnosis = ''
            
            inputData = [Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Albumin]
            
            # creating buttons for Prediction and Clear
            c1, c2, c3 = st.columns([3, 1, 1],gap="large")
            
            with c1:
                submitted = st.form_submit_button(label="Liver Disease Test Result")
            with c2:
                pass
            with c3:
                clear = st.form_submit_button(label="Clear",on_click=clear_form,args=("liver", ))
            
            
            if submitted:
                if(valid_input(inputData)):   
                    liver_prediction = liver_model.predict([[Total_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Albumin]])
                        
                    if (liver_prediction[0] == 0 or liver_prediction[0] == 2):
                        liver_diagnosis = 'The person does not have liver disease'
                        st.success(liver_diagnosis)
                    else:
                        liver_diagnosis = 'The person has some liver disease'
                        st.error(liver_diagnosis)
                else:
                    st.error("All fields are required and must have appropriate values")
        
    # Kidney Disease Prediction Page --- used rf model : 97.5% accuracy

    # if (selected == 'Kidney Disease Prediction'):
    with tab4:
        #page title
        st.title('Kidney Disease Prediction') 

        submitted = False
        clear = False
        
        with st.form(key="kidneyForm"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                Albumin = st.text_input('Albumin',key='AlbuminK')
            
            with col2:
                Blood_Urea = st.text_input('Blood Urea (mgs/dL)',key='Blood_Urea')
            
            with col3:
                Serum_Creatinine = st.text_input('Serum Creatinine (mgs/dL)',key='Serum_Creatinine')
                
            with col1:
                Sodium = st.text_input('Sodium (mEq/L)',key='Sodium')
                
            with col2:
                Potassium = st.text_input('Potassium (mEq/L)',key='Potassium')
        
            # code for Prediction
            kidney_diagnosis = ''
            
            inputData = [Albumin,Blood_Urea,Serum_Creatinine,Sodium,Potassium]
            
            # creating buttons for Prediction and Clear
            c1, c2, c3 = st.columns([3, 1, 1],gap="large")
            
            with c1:
                submitted = st.form_submit_button(label="Kidney Disease Test Result")
            with c2:
                pass
            with c3:
                clear = st.form_submit_button(label="Clear",on_click=clear_form,args=("kidney", ))
            
            
            if submitted:
                if(valid_input(inputData)):   
                    kidney_prediction = kidney_model.predict([inputData])
                        
                    if (kidney_prediction[0] == 0):
                        kidney_diagnosis = 'The person does not have kidney disease'
                        st.success(kidney_diagnosis)
                    else:
                        kidney_diagnosis = 'The person has some kidney disease'
                        st.error(kidney_diagnosis)
                else:
                    st.error("All fields are required and must have appropriate values")

    # =============================================================================

if (selected == 'Details fetched from CSV File'):

    # Flags
    flagKidney, flagLiver, flagHeart, flagDiab = 0,0,0,0

    uploaded_file = st.file_uploader('Upload the Blood Report:')

    # filename = 
    data = []
    temp = []
    columns = []
    Rd = pd.DataFrame(data)

    # Report Generation

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        btReport = df.columns

        for i in kidneyKeys:
            if i not in btReport:
                flagKidney = 1
                break

        for i in liverKeys:
            if i not in btReport:
                flagLiver = 1
                break

        for i in heartKeys:
            if i not in btReport:
                flagHeart = 1
                break

        for i in diabKeys:
            if i not in btReport:
                flagDiab = 1
                break

        countRows = len(df.index)

        for i in range(0, countRows):
            if flagKidney == 0:
                dfK = df[['AlbuminK','Blood_Urea','Serum_Creatinine','Sodium','Potassium']]
                inp = dfK.loc[i:i+1]
                kidney_prediction = kidney_model.predict(inp.values)

                # print(inp.values)

                if (kidney_prediction[0] == 0):
                    kidney_diagnosis = 'NO'
                else:
                    kidney_diagnosis = 'YES'
                
                # print(kidney_prediction)
                # print("Kidney: "+kidney_diagnosis)
                # data.append(kidney_diagnosis)
                Rd.at[i, "Kidney Disease"] = kidney_diagnosis

            if flagLiver == 0:
                dfK = df[['Total_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','AlbuminL']]
                inp = dfK.loc[i:i+1]
                liver_prediction = liver_model.predict(inp.values)
                        
                if (liver_prediction[0] == 0 or liver_prediction[0] == 2):
                        liver_diagnosis = 'NO'
                else:
                    liver_diagnosis = 'YES'
                
                # print("Liver: "+liver_diagnosis)
                Rd.at[i, "Liver Disease"] = liver_diagnosis

            if flagHeart == 0:
                dfK = df[['age','sex','cp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
                inp = dfK.loc[i:i+1]
                heart_prediction = heart_model.predict(inp.values)
                        
                if (heart_prediction[0] == 0):
                        heart_diagnosis = 'NO'
                else:
                    heart_diagnosis = 'YES'
                
                # print("Heart: "+heart_diagnosis)
                Rd.at[i, "Heart Disease"] = heart_diagnosis

            if flagDiab == 0:
                dfK = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
                inp = dfK.loc[i:i+1]
                diab_prediction = diabetes_model.predict(inp.values)
                        
                if (diab_prediction[0] == 0):
                    diab_diagnosis = 'NO'
                else:
                    diab_diagnosis = 'YES'
                
                # print("Diabetes: "+diab_diagnosis)
                Rd.at[i, "Diabetes"] = diab_diagnosis

        st.dataframe(Rd)

    # ----------------------------

    RdT = Rd
    rows = len(RdT.index)

    csv = convert_df(Rd)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Prediction Report",
            data=csv,
            file_name='Pred_Report.csv',
            mime='text/csv',
            key = "csv"
        )

    # with col2:
    #     download("", "EmptyFile.txt", "text")
    #     if not RdT.empty:
    #         for i in range(0, 3):
    #             eachRow = RdT.iloc[i:i+1]
    #             content = ' '.join([str(elem) for elem in eachRow])
    #             fn = 'Pred_Report_Patient_'+ str(i+1) +'.txt'
    #             key = 'text'+ str(i+1)
    #             download(content, fn, key)
