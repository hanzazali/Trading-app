#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:16:03 2023

@author: hanzazali
"""

#import relevant librariesc

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import yfinance as yf


#import dataset
df = yf.download('EURUSD=X')

df.drop(['Adj Close', 'Volume'], axis = 1, inplace = True)


#create the starting date for predictions
# create a starting date
from datetime import datetime, timedelta

if datetime.today().date().weekday()>4:
    start = datetime.today().date() + timedelta(days = 2)
    
else:
    start = datetime.today().date() - timedelta(days = datetime.today().date().weekday())


#features extraction function
def preprocess_multistep_lstm(sequence, n_steps_in, n_steps_out, n_features):
    
    X, y = list(), list()
    
    for i in range(len(sequence)):
        
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
            
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix, -1:]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    return X, y


#create prediction features
# Number of days into the future we want to predict
n_steps_out = 5

# choose the number of days on which to base our predictions 
nb_days = 20

n_features = 4

inputs, target = preprocess_multistep_lstm(df.tail(25).to_numpy(), nb_days, n_steps_out, n_features)


#load the model
def load_model():
    with open('Model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()


#make predictions
predictions = pd.DataFrame(model.predict(inputs).T, 
                          index = [start + timedelta(days = i) for i in range(5)],
                          columns = ['pred Close'])


#display predictions
print(predictions)


fig, ax = plt.subplots()
ax.plot(predictions.index, predictions['pred Close'])
ax.set_xlabel('date')
ax.set_ylabel('price')

plt.xticks(rotation = 90)
plt.show()



#historical price


fig2 = px.line(df, x=df.index, y="Close")
fig.show()


def main():
    
    st.title('EUR/USD weekly predictions')
    
    st.header('weekly predictions display')
    st.pyplot(fig)
    
    st.header('weekly predictions table')
    st.write(round(predictions, 6))
    
    st.header('historical price')
    st.plotly_chart(fig2)
    
     
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



