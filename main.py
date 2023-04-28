import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

data = pd.read_csv('diabetes_prediction_dataset.csv')
#check for null values
print(data.isnull().sum())

X = data.drop('diabetes',axis=1)
y=data['diabetes']

feature_num = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
feature_cat = ['gender', 'smoking_history']

transformer = ColumnTransformer(transformers=[('ss',StandardScaler(),feature_num),
                                                ('ohe', OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=True), feature_cat)],remainder='passthrough')

my_pipeline = Pipeline(steps=[('preprocessor', transformer)
                             ])
processed_X = my_pipeline.fit_transform(X)

X_train, X_test, y_train , y_test = train_test_split(processed_X, y, test_size = 0.3, random_state = 42)

model = LogisticRegression(random_state = 42)
model.fit(X_train, y_train)

y_train_repo = model.predict(X_train)
y_test_repo = model.predict(X_test)
print(f"the accuracy on train set {accuracy_score(y_train, y_train_repo)}")
print(f"the accuracy on test set {accuracy_score(y_test, y_test_repo)}")