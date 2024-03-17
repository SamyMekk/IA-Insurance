

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler



st.markdown("<h1 style='text-align: center; color: black;'>Approche Machine Learning par Random Forest : Prédiction Variable Response</h1>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align : center ; color: black;'> On définit ci-dessous les métriques qui vont juger de la qualité du modèle :  </h3>", unsafe_allow_html=True)


st.latex("Accuracy=\\frac{TP+TN}{TP+FP+TN+FN}")
st.latex("Precision=\\frac{TP}{TP+FP}")
st.latex("Recall=\\frac{TP}{TP+FN}")
st.latex("BalancedAccuracyScore=\\frac{1}{2}(\\frac{TP}{TP+FN} + \\frac{TN}{TN+FP})")
st.latex("F1-Score=\\frac{2 \\times Precision \\times Recall}{Precision + Recall }")


st.markdown("<h5 style='text-align: center; color: black;'>Avec :</h5>", unsafe_allow_html=True)

st.markdown("<h6 style='text-align : center ; color: black;'>TN : True Negatives, TP : True Positives, FN: False Negatives et FP : False Positive </h6>",unsafe_allow_html=True)





# Loading Data
def load_data(url):
    df=pd.read_csv(url)
    return df

train_data=load_data(r"C:\Users\samym\OneDrive\Bureau\ENSAE 3A\IA_Assurance\ensae_ai_insurance\Projet\dataset\train.csv")


train_data = train_data.drop(['id'], axis=1)


train_data['Gender'] = train_data['Gender'].map({'Male': 1, 'Female': 0})
train_data['Vehicle_Damage'] = train_data['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
train_data['Vehicle_Age'] = train_data['Vehicle_Age'].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})


# Feature Engineering 


def print_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    bal=balanced_accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true, y_pred)
    data={'''Accuracy''': acc,
         '''Recall''':rec,
         '''Precision''':pre,
         '''Balanced''':bal,
         '''F1 Score''':f1}
    Parametres=pd.DataFrame(data,index=[0])
    return Parametres


X = train_data.drop(['Response'], axis=1).values
y = train_data['Response'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sampler = RandomUnderSampler()
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)

if st.button("Lancez l'analyse du modèle "):
    forest = RandomForestClassifier(n_estimators=100, max_depth=3)
    forest.fit(X_train_resampled, y_train_resampled)
    y_forest = forest.predict(X_test_scaled)
    tn_forest, fp_forest, fn_forest, tp_forest = confusion_matrix(y_test, y_forest).ravel()
    st.markdown("<h3 style='text-align: center; color: black;'>Voici le résultat de la matrice de confusion :</h3>", unsafe_allow_html=True)
    Dataset=pd.DataFrame({"TN":tn_forest,"FP":fp_forest,"FN":fn_forest,"TP":tp_forest},index=[0])
    st.write(Dataset)
    Result=print_metrics(y_test, y_forest).round(2)
    st.markdown("<h3 style='text-align: center; color: black;'>Voici les métriques de notre modèle :</h3>", unsafe_allow_html=True)
    st.write(Result)



  

#final_params = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
#                'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 
#                'min_impurity_decrease': 0.0, 'min_samples_leaf': 4, 'min_samples_split': 5,
#                'min_weight_fraction_leaf': 0.0, 'n_estimators': 50, 'n_jobs': None, 'oob_score': False,
##                'random_state': 42, 'verbose': 0, 'warm_start': False}

#best_model = RandomForestClassifier(**final_params).fit(X_train, y_train)

# Obtenez les prédictions du meilleur modèle sur l'ensemble de test

# Créer la matrice de confusion



# Évaluez la performance du modèle sur l'ensemble de test





