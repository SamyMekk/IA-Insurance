# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:39:26 2024

@author: samym
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Application Streamlit pour le projet d'IA pour l'Actuariat")

st.subheader("On va s'intéresser au dataset suivant dont on donne ci-dessous les premières lignes")

@st.cache_data
def load_data(url):
    df=pd.read_csv(url)
    return df

dataset=load_data("https://raw.githubusercontent.com/SamyMekk/IA-Insurance/master/train.csv?token=GHSAT0AAAAAACN4BSHDWIBDDVUB2AAL2MEAZOLPXLA",index_col=0)

st.write(dataset.head(5))

st.subheader("On va s'intéresser aux différentes covariates qui pourraient intervenir dansa le calcul du montant de la prime ")



def user_input():
    pass



grouped_dataage = dataset.groupby('Age')['Annual_Premium'].mean()


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(grouped_dataage.index, grouped_dataage.values, marker='o', linestyle='-')
ax.set_ylabel('Montant moyen de la prime d\'assurance')
ax.set_title('Montant moyen de la prime d\'assurance en fonction de l\'âge de l\'assureur')
ax.set_xlabel('Âge de l\'assureur')
ax.grid(True)
st.pyplot(fig)

fig2,ax2=plt.subplots(1,1,figsize=(10,10))
filtered_data = dataset[(dataset['Age'] >= 20) & (dataset['Age'] <= 30)]

# Tracer la densité des primes annuelles
sns.kdeplot(filtered_data['Annual_Premium'], shade=True)
ax2.set_xlabel('Montant de la prime annuelle')
ax2.set_ylabel('Densité')
ax2.set_title('Densité des primes annuelles pour des assurés âgés de 20 à 30 ans')
ax2.grid(True)
st.pyplot(fig2)
# Créer le graphique


print("done")


# Quelques Plots pour des graphiques en EDA

