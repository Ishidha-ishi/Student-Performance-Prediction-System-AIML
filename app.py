import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

st.title('Student Performance Prediction System')

def load_and_train():
    df = pd.read_csv('Dataset.csv')
    X = df[["Student_Attendance_Percentage", "Study_Hours_Per_Day", "Learning_Activities_Score"]]
    y = df["Final_Performance_Percentage"]
    
    # Create pipeline and fit
    pipeline = make_pipeline(SimpleImputer(strategy="mean"), LinearRegression())
    pipeline.fit(X, y)
    return pipeline

model = load_and_train()

attendance = st.number_input('Student Attendance Percentage', min_value=0.0, max_value=100.0)
study_hours = st.number_input('Study Hours Per Day', min_value=0.0, max_value=24.0)
learning_activities = st.number_input('Learning Activities Score', min_value=0.0, max_value=10.0)

is_valid = True

if attendance < 0 or attendance > 100:
    st.error("Attendance percentage must be between 0 and 100.")
    is_valid = False

if study_hours < 0 or study_hours > 24:
    st.error("Study hours per day must be between 0 and 24.")
    is_valid = False

if learning_activities < 0 or learning_activities > 10:
    st.error("Learning activities score must be between 0 and 10.")
    is_valid = False

print(is_valid)

if st.button('Predict'):
    if is_valid:
        input_data = pd.DataFrame({
            "Student_Attendance_Percentage": [attendance],
            "Study_Hours_Per_Day": [study_hours],
            "Learning_Activities_Score": [learning_activities]
        })
        
        prediction = model.predict(input_data)
        final_score = max(0.0, min(100.0, prediction[0]))
        st.success(f'Predicted Final Performance Percentage: {final_score:.2f}%')
    else:
        st.error("Please fix the validation errors above before predicting.")
