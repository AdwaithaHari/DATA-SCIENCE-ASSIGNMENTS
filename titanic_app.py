import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Load the Titanic Dataset
titanic_deployment = pd.read_csv("C:\\Users\\adwai\\Downloads\\Deployment\\Titanic_deployment_data.csv")

# Create target and features
target_column = ['Survived']
feature_column = ['PassengerId','Pclass','Sex','Age','Fare',
                  'Embarked','SibSp','Parch']     # specify features

# Separate features (X) and target (Y)
X = titanic_deployment[feature_column]
Y = titanic_deployment[target_column]

# Train Model

model = LogisticRegression()
model.fit(X,Y)

# Streamlit App for Predictions
st.title('Titanic Survival Prediction')
st.sidebar.header('Enter Passenger Details')

def user_input_features():
  PassengerId = st.sidebar.number_input("Enter the Passenger Id", min_value = 1, step = 1)
  Pclass = st.sidebar.selectbox("Enter the Passenger class", ("1","2","3"))
  Sex = st.sidebar.selectbox("Enter the Sex('Male = 1', 'Female = 0')", ("0","1"))
  Age = st.sidebar.number_input("Enter the  Age", min_value = 0.0, step = 1.0)
  Fare = st.sidebar.number_input("Enter the  Fare", min_value = 0.0, step = 1.0)
  Embarked = st.sidebar.selectbox("Enter the Embarked Place ('Cherbourg = 0', 'Queenstown = 1','Southampton = 2')",("0","1","2"))
  SibSp = st.sidebar.selectbox("Enter the number of Siblings/Spouse Aboard",("0","1","2","3","4","5","8"))
  Parch = st.sidebar.selectbox("Enter the number of Parents/Children Aboard", ("0","1","2","3","4","5","6","9"))

  # Construct the dataframe
  data = {'PassengerId':PassengerId,
          'Pclass':Pclass,
          'Sex':Sex,
          'Age':Age,
          'Fare':Fare,
          'Embarked':Embarked,
          'SibSp':SibSp,
          'Parch':Parch}
  return pd.DataFrame(data, index = [0])

# Get User inputs
df = user_input_features()

# Display the user input features
st.subheader('User Input features')
st.write(df)

# Make Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Display Prediction
st.subheader('Survival Prediction Analysis')
st.write('Prediction: Survived' if prediction_proba[0][1] > 0.5 else 'Prediction: Not Survived')
st.write(f'Surviving Probability: {prediction_proba[0][1]*100:.2f}%')
st.write(f'Not Surviving Probability: {prediction_proba[0][0]*100:.2f}%')

