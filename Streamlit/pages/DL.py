# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:26:20 2024

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


# Visualizing 
from tqdm import tqdm

# Deep Learning
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



st.markdown("<h1 style='text-align: center; color: black;'>Approche Deep Learning : Prédiction Variable Response</h1>", unsafe_allow_html=True)


# Feature Engineering 


# Loading Data
def load_data(url):
    df=pd.read_csv(url)
    return df

train_data=load_data(r"C:\Users\samym\OneDrive\Bureau\ENSAE 3A\IA_Assurance\ensae_ai_insurance\Projet\dataset\train.csv")

train_data = train_data.drop(['id'], axis=1)

train_data['Gender'] = train_data['Gender'].map({'Male': 1, 'Female': 0})
train_data['Vehicle_Damage'] = train_data['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
train_data['Vehicle_Age'] = train_data['Vehicle_Age'].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})


def remove_outliers(df, col_name):
    q25 = df[col_name].quantile(0.25)
    q75 = df[col_name].quantile(0.75)
    
    cut_off = 1.5 * (q75 - q25)
    lower, upper = max(0, q25 - cut_off), q75 + cut_off
    
    df.loc[df[col_name] > upper, col_name] = upper
    df.loc[df[col_name] < lower, col_name] = lower
    
    return df[col_name]

train_data['Annual_Premium'] = remove_outliers(train_data, 'Annual_Premium')



# Training consists of gradient steps over mini batch of data
def train(model, trainloader, criterion, optimizer, epoch, num_epochs):
    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    loop = tqdm(trainloader)
#     loop.set_description(f'Training Epoch [{epoch + 1}/{num_epochs}]')

    # We iterate over the mini batches of our data
    for inputs, targets in loop:

        optimizer.zero_grad() # Erase any previously stored gradient
        outputs = model(inputs) # Forwards stage (prediction with current weights)
        targets = targets.view(-1, 1) # Reshape targets to match the output size
        loss = criterion(outputs, targets) # loss evaluation
        loss.backward() # Back propagation (evaluate gradients)

        # Making gradient step on the batch (this function takes care of the gradient step for us)
        optimizer.step()

def validation(model, valloader, criterion):
    # Do not compute gradient, since we do not need it for validation step
    with torch.no_grad():
        
        model.eval() # We enter evaluation mode.

        total, correct = 0, 0 # Keep track of currently used samples
        running_loss = 0.0 # Accumulated loss without averaging

        loop = tqdm(valloader) # This is for the progress bar
#         loop.set_description('Validation in progress')

        # We again iterate over the batches of validation data. batch_size does not play any role here
        for inputs, targets in loop:
            outputs = model(inputs) # Run samples through our net
            total += inputs.shape[0] # Total number of used samples
            predicted = torch.round(outputs)
            targets = targets.view(-1, 1)
            correct += (predicted == targets).sum().item()
            
            # Multiply loss by the batch size to erase averagind on the batch
            batch_loss = criterion(outputs, targets).item()
            running_loss += batch_loss * inputs.shape[0]
            
#             loop.set_postfix(val_loss=(running_loss / total), val_acc=(correct / total)) # Set nice progress message
        return running_loss / total, correct / total


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_prob=0.5):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout layer after first hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout layer after second hidden layer
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.sigmoid = nn.Sigmoid()  # Use sigmoid activation for binary classification

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)  # Applying dropout after first hidden layer
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)  # Applying dropout after second hidden layer
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
    
   
    
   
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
    
# Convert data to PyTorch DataLoader
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # No need to shuffle for validation


def user_input():
    epochs=st.sidebar.number_input("Choississez le nombre d'epochs pour l'entraînement du modèle", 5,10 )
    hidden_sizes1=st.sidebar.number_input("Choississez le nombre de neurones pour la 1ère couche ",32,128)
    hidden_sizes2=st.sidebar.number_input("Choississez le nombre de neurones pour la 2ème couche ",64,128)
    dropout_prob=st.sidebar.number_input("Choississez la dropout probabilité ",0.05,0.2)
    learning_rate=st.sidebar.number_input("Choississez le learning rate",0.001,0.01)
    
    return epochs, hidden_sizes1, hidden_sizes2, dropout_prob, learning_rate
    
    
input_size=10
criterion = nn.BCELoss()

epochs,hidden_sizes1,hidden_sizes2,dropout_prob,learning_rate=user_input()

Infos=pd.DataFrame({"Nombre d'epochs":epochs,"Hidden Size 1": hidden_sizes1,"Hidden Size 2": hidden_sizes2,
              "Dropout ":dropout_prob,"Learning Rate":learning_rate},index=[0])


st.markdown("<h3 style='text-align: center; color: black;'>Voici les paramètres que vous avez sélectionné pour votre réseau de neurones : </h3>", unsafe_allow_html=True)

st.write(Infos,use_container_width=True)



st.markdown("<h3 style='text-align: center; color: black;'>Mathématiquement, celà se réécrit :</h3>", unsafe_allow_html=True)


st.latex("y=\Psi(W_{out}\phi(W_{2}(\phi(W_1 x + b_1)+b_2)+b_{out}) ")



st.markdown("<h5 style='text-align: center; color: black;'>Avec :</h5>", unsafe_allow_html=True)


st.latex("\Psi(x)=\\frac{1}{1+e^{-x}}")
st.latex("\phi(x) =x^{+}")
st.latex("W_{{1}} \in M_{{ {} \\times {} }} , W_{{2}} \in M_{{ {} \\times {} }},  W_{{out}} \in M_{{ {} \\times {} }}".format(hidden_sizes1, input_size,hidden_sizes2,hidden_sizes1,hidden_sizes2,1))





# \varphi\left(W_2 \varphi\left(W_1 x + b_1\right) + b_2\right)+b_{out})")

hidden_sizes=[hidden_sizes1,hidden_sizes2]


ffn_model = FeedForwardNet(input_size, hidden_sizes, dropout_prob)
ffn_optimizer = optim.Adam(ffn_model.parameters(), lr=learning_rate)




if st.button("Lancez l'entraînement et l'analyse du modèle"):
    for epoch in range(epochs):
        #makes one pass over the train data and updates weights
        train(ffn_model, train_loader, criterion, ffn_optimizer, epoch, epochs)
        # makes one pass over validation data and provides validation statistics
        val_loss, val_acc = validation(ffn_model, val_loader, criterion)
    ffn_model.eval()
    with torch.no_grad():
        y_ffn = ffn_model(X_test_tensor)
    tn, fp, fn, tp = confusion_matrix(y_test, y_ffn.round()).ravel()
    Dataset=pd.DataFrame({"TN":tn,"FP":fp,"FN":fn,"TP":tp},index=[0])
    
    st.markdown("<h3 style='text-align: center; color: black;'>Voici le résultat de la matrice de confusion :</h3>", unsafe_allow_html=True)

    st.write(Dataset)
    acc = accuracy_score(y_test, y_ffn.round())
    rec = recall_score(y_test, y_ffn.round())
    pre = precision_score(y_test, y_ffn.round())
    f1 = f1_score(y_test, y_ffn.round()).round(2)
    balanced=balanced_accuracy_score(y_test,y_ffn.round())
    Result=print_metrics(y_test, y_ffn.round()).round(2)
    st.markdown("<h3 style='text-align: center; color: black;'>Voici les métriques de notre modèle :</h3>", unsafe_allow_html=True)
    st.write(Result)


    
    
    
    
    
