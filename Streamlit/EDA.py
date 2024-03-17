# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:39:26 2024

@author: samym
"""

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
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical



st.markdown("<h1 style='text-align: center; color: black;'>Projet IA pour l'Actuariat </h1>", unsafe_allow_html=True)


st.markdown("<h3 style='color: black;'>Une vue d'ensemble du dataset :  </h3>", unsafe_allow_html=True)


#@st.cache_data
def load_data(url):
    df=pd.read_csv(url)
    return df

train=load_data(r"C:\Users\samym\OneDrive\Bureau\ENSAE 3A\IA_Assurance\ensae_ai_insurance\Projet\dataset\train.csv")
test=load_data(r"C:\Users\samym\OneDrive\Bureau\ENSAE 3A\IA_Assurance\ensae_ai_insurance\Projet\dataset\test.csv")

st.write(train.head(5))

st.markdown("<h3 style='color: black;'>Exploratory Data Analysis :  </h3>", unsafe_allow_html=True)



def user_input():
    pass



# Code

selected_chart = st.sidebar.selectbox("Sélectionnez la variable d'intérêt pour voir sa répartition :", ("Gender", "Driving_License","Previously_Insured","Vehicle_Damage","Vehicle_Age","Response"))


fig = make_subplots(rows=1, cols=2)

if selected_chart == "Gender":
   # Création du graphique Train Gender
    trace_train = go.Bar(
        x=['Male', 'Female'], 
        y=[
            len(train[train[selected_chart]=='Male']),
            len(train[train[selected_chart]=='Female'])
        ], 
        name='Train Gender',
        text=[
            str(round(100 * len(train[train[selected_chart]=='Male']) / len(train), 2)) + '%',
            str(round(100 * len(train[train[selected_chart]=='Female']) / len(train), 2)) + '%'
        ],
        textposition='auto'
    )
    fig.append_trace(trace_train, 1, 1)
    fig.update_xaxes(title_text=selected_chart, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
elif selected_chart == "Driving_License" or selected_chart=="Previously_Insured" or selected_chart=="Response":
    # Création du graphique Test Gender
    trace_test = go.Bar(
        x=[1, 0], 
        y=[
            len(train[train[selected_chart]==1]),
            len(train[train[selected_chart]==0])
        ], 
        name='Test Gender',
    )
    fig.append_trace(trace_test, 1, 1)
    fig.update_xaxes(title_text=selected_chart, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    
elif selected_chart == "Vehicle_Damage":
    # Création du graphique Test Gender
    trace_test = go.Bar(
        x=['Yes ', 'No'], 
        y=[
            len(train[train[selected_chart]=="Yes"]),
            len(train[train[selected_chart]=="No"])
        ], 
        name='Test Gender',
        text=[
            str(round(100 * len(test[test[selected_chart]=="Yes"]) / len(test), 2)) + '%',
            str(round(100 * len(test[test[selected_chart]=="No"]) / len(test), 2)) + '%'
        ],
        textposition='auto'
    )
    fig.append_trace(trace_test, 1, 1)
    fig.update_xaxes(title_text=selected_chart, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    

    
elif selected_chart == "Vehicle_Age":
    # Création du graphique Test Gender
    trace_test = go.Bar(
        x=['> 2 Years','1-2 Year', '< 1 Year'], 
        y=[
            len(train[train[selected_chart]=='> 2 Years']),
            len(train[train[selected_chart]=='1-2 Year']),
            len(train[train[selected_chart]=='< 1 Year'])
        ], 
        name='Test Gender',
        text=[
            str(round(100 * len(test[test[selected_chart]=='> 2 Years']) / len(test), 2)) + '%',
            str(round(100 * len(test[test[selected_chart]=='1-2 Year']) / len(test), 2)) + '%',
            str(round(100 * len(test[test[selected_chart]=='< 1 Year']) / len(test), 2)) + '%'
        ],
        textposition='auto'
    )
    fig.append_trace(trace_test, 1, 1)
    fig.update_xaxes(title_text=selected_chart, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    


# Mise à jour de la mise en page du graphique
fig.update_layout(
    title_text=selected_chart,
    height=400,
    width=700
)

# Affichage du graphique dans Streamlit
st.plotly_chart(fig)




if st.button("Quelques Graphiques à afficher :"):
    plot2= sns.heatmap(train.corr().round(2),cmap="viridis",annot=True)
    st.pyplot(plot2.get_figure())
    grouped_dataage = train.groupby('Age')['Annual_Premium'].mean()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(grouped_dataage.index, grouped_dataage.values, marker='o', linestyle='-')
    ax.set_ylabel('Montant moyen de la prime d\'assurance')
    ax.set_title('Montant moyen de la prime d\'assurance en fonction de l\'âge de l\'assureur')
    ax.set_xlabel('Âge de l\'assureur')
    ax.grid(True)
    st.pyplot(fig)
    fig2,ax2=plt.subplots(1,1,figsize=(10,10))
    filtered_data = train[(train['Age'] >= 20) & (train['Age'] <= 30)]
    # Tracer la densité des primes annuelles
    sns.kdeplot(filtered_data['Annual_Premium'], shade=True)
    ax2.set_xlabel('Montant de la prime annuelle')
    ax2.set_ylabel('Densité')
    ax2.set_title('Densité des primes annuelles pour des assurés âgés de 20 à 30 ans')
    ax2.grid(True)
    st.pyplot(fig2)
    
    
selected_chartinfo = st.sidebar.selectbox("Sélectionnez la variable d'intérêt pour voir son comportement vis-à-vis de la variable Response:", ("Gender", "Driving_License","Previously_Insured","Vehicle_Damage","Vehicle_Age"))
plot=sns.countplot(train,x=selected_chartinfo,hue="Response")
plt.title("Etude du lien entre la variable {} et la target Response".format(selected_chartinfo))
st.pyplot(plot.get_figure())





# Quelques Plots pour des graphiques en EDA

