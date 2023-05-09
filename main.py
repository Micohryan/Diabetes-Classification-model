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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

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

lrc = LogisticRegression()

knc = KNeighborsClassifier()
dtc = DecisionTreeClassifier(random_state=2)
lrc = LogisticRegression()
rfc = RandomForestClassifier(random_state=2)
abc = AdaBoostClassifier(random_state=2)
bc = BaggingClassifier(random_state=2)
etc = ExtraTreesClassifier(random_state=2)
gbdt = GradientBoostingClassifier(random_state=2)
xgb = XGBClassifier(random_state=2,use_label_encoder=False,eval_metric='mlogloss')


clfs = {
    'KN' : knc,
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,zero_division=0)
    
    return accuracy,precision

accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
#     print("For ",name)
#     print("Accuracy - ",current_accuracy)
#     print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
print(performance_df)