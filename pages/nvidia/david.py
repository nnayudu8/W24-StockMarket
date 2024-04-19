# -*- coding: utf-8 -*-
"""David's SMA 2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XVrgkrjDO-WGIkMRtT-9xVtaQ3J2FKAl
"""

# Commented out IPython magic to ensure Python compatibility.
#!pip install pandas_ta

#to track execution times for each cell
#!pip install ipython-autotime
# %load_ext autotime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
import streamlit as st

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
import st_pages
from st_pages import Page, show_pages, Section, add_page_title

st.markdown("""
<style>
	[data-testid="stHeader"] {
		background-image: linear-gradient(90deg, rgb(22, 230, 48), rgb(112, 128, 144));
	}
</style>""",
unsafe_allow_html=True)

show_pages(
    [
        Page("home.py", "Home", "🏠"),
        Section(name="Research Results", icon=":computer:"),
        Page("pages/indicators.py", "Technical Indicators", ":money_with_wings:"),
        Section(name="Model Results", icon=":chart_with_upwards_trend:"),
        Page("pages/base/base_model.py", "Base Model", ":dollar:"),
        Page("pages/david/david.py", "David", ":euro:"),
        Page("pages/david/presentation.py", "Shayan", ":pound:"),
        Page("pages/soon.py", "Coming Soon...", ":eyes:"),
    ]
)

st_pages.add_indentation()

def main():

    data = yf.download(tickers = 'NVDA')

    data['RSI']=ta.rsi(data.Close, length=15)
    data['EMAF']=ta.ema(data.Close, length=20)
    data['EMAM']=ta.ema(data.Close, length=100)
    data['EMAS']=ta.ema(data.Close, length=150)

    ###ADD MORE INDICATORS###
    #OBV = on-balance volume
    data['OBV']=ta.obv(data.Close, data.Volume)

    #ADX = avg directional movement index
    #adx_df = ta.adx(data['High'], data['Low'], data['Close'])
    #print(adx_df.columns)
    adx_df = ta.adx(data.High, data.Low, data.Close)
    data['ADX'] = adx_df['ADX_14']  # The ADX line
    data['DMP'] = adx_df['DMP_14']  # The +DI line
    data['DMN'] = adx_df['DMN_14']  # The -DI line

    #STOCH = stochastic oscillator
    #stoch_df = ta.stoch(data['High'], data['Low'], data['Close'])
    #print(stoch_df.columns)
    stoch_df = ta.stoch(data.High, data.Low, data.Close)
    data['STOCH'] = stoch_df['STOCHk_14_3_3']  # The STOCH line
    #data['???'] = stoch_df['???']  # The ??? line

    #MACD = moving avg convergence divergence
    macd_df = ta.macd(data.Close)
    #print(macd_df.columns)
    data['MACD'] = macd_df['MACD_12_26_9']

    #BBANDS = bollinger bands
    bbands_df = ta.bbands(data.Close)
    data['BBL'] = bbands_df['BBL_5_2.0'] #lower
    data['BBM'] = bbands_df['BBM_5_2.0'] #middle
    data['BBU'] = bbands_df['BBU_5_2.0'] #upper


    ###DATASET INDEXING###
    data['Target'] = data['Adj Close']-data.Open
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

    data['TargetNextClose'] = data['Adj Close'].shift(-1)

    data.dropna(inplace=True)

    # Filter the dataset for dates from 2018 onwards
    data_set = data[data.index.year >= 2018]


    data.reset_index(inplace = True)

    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

    target_col = "TargetNextClose"

    from sklearn.preprocessing import MinMaxScaler
    sc_data = MinMaxScaler(feature_range=(0,1))
    sc_response = MinMaxScaler(feature_range=(0,1))

    response_scaled = sc_response.fit_transform(data_set[[target_col]])
    # print(response_scaled)
    # print(data_set.loc[:, data_set.columns != target_col])
    # print(response_scaled)
    data_set_scaled = sc_data.fit_transform(data_set.loc[:, data_set.columns != target_col])
    print(data_set_scaled.shape)
    # print(data_set_scaled)
    print(len(data_set_scaled))
    data_set_scaled_new = []
    for i in range(len(data_set_scaled)):
        data_set_scaled_new.append([])
        # print(response_scaled[i][0])

        # print(data_set_scaled[i])
        data_set_scaled_new[i] = np.append(data_set_scaled[i],response_scaled[i])
        # print(data_set_scaled_new[i])
        # print("#"*80)

    data_set_scaled_new = np.asarray(data_set_scaled_new)
    print(data_set_scaled_new)

    # multiple feature from data provided to the model
    X = []
    #print(data_set_scaled[0].size)
    #data_set_scaled=data_set.values
    backcandles = 30
    print(data_set_scaled_new.shape[0])
    for j in range(15):#data_set_scaled_new[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled_new.shape[0]):#backcandles+2
            X[j].append(data_set_scaled_new[i-backcandles:i, j])

    #move axis from 0 to position 2
    X=np.moveaxis(X, [0], [2])
    #Erase first elements of y because of backcandles to match X length
    #del(yi[0:backcandles])
    #X, yi = np.array(X), np.array(yi)
    # Choose -1 for last column, classification else -2...
    X, yi =np.array(X), np.array(data_set_scaled_new[backcandles:,-1])
    y=np.reshape(yi,(len(yi),1))
    #y=sc.fit_transform(yi)
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # split data into train test sets
    splitlimit = int(len(X)*0.8)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]

    
    #tf.random.set_seed(20)
    model = keras.models.load_model('davidStock.keras')

    y_pred = model.predict(X_test)
    #y_pred=np.where(y_pred > 0.43, 1,0)
    for i in range(10):
        print(y_pred[i], y_test[i])

    print(sc_response.inverse_transform(y_test)[0])
    print(sc_response.inverse_transform(y_pred)[0])

    plt.figure(figsize=(16,9))
    plt.plot(y_test, color = 'black', label = 'Test')
    plt.plot(y_pred, color = 'green', label = 'pred')
    plt.legend()

    st.header("Graph of stock price vs predicted stock price")
    st.pyplot(plt)
    #add RMSE and MAPE calculations
    #RMSE = Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print("RMSE: ", end="")
    print(rmse)


    #MAPE = Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print("MAPE (%): ", end="")
    print(mape)

if __name__ == "__main__":
    main()

