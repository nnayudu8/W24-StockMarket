import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime
# import plotly.graph_objects as go
import pandas_ta as ta
import streamlit as st
import tensorflow as tf
import keras
import st_pages
from st_pages import Page, show_pages, Section, add_page_title
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Ensure GPU is not used

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
        Page("pages/nvidia/nvidia.py", "NVIDIA", ":euro:"),
        Page("pages/nvidia/nvidia2.py", "NVIDIA v2", ":pound:"),
        Page("pages/soon.py", "Coming Soon...", ":eyes:"),
    ]
)

st_pages.add_indentation()

def main():
    cc1, cc2 = st.columns([0.8, 0.2])
    with cc1:
        st.title("NVIDIA Daily Trend Prediction")

    with cc2:
        st.image("images/nvidia_logo.png", width=175)

    st.write(
        '''
        ***
        NVIDIA has been a hot company and we want to hop on that train! We set out to find the most relevant factors in predicting
        the probability that the price will increase or decrease in the next close so that we know what factors are important. 

        ⇨ Shayan Sinha and David Sanico
        ***
    '''
    )

    data = yf.download(tickers = 'NVDA')
    data.columns = data.columns.droplevel(1)
    df = data.copy()
    df.reset_index(inplace = True)
    extrema = 0.05
    for i in range(8,len(df)-8):
        if(((df["Close"].iloc[i-7]-df["Close"].iloc[i]) / df["Close"].iloc[i]) > extrema) and (((df["Close"].iloc[i+7]-df["Close"].iloc[i]) / df["Close"].iloc[i]) > extrema):
            df.loc[i,"minmax"] = -1
        elif (((df["Close"].iloc[i]-df["Close"].iloc[i+7]) / df["Close"].iloc[i]) > extrema) and (((df["Close"].iloc[i]-df["Close"].iloc[i-7]) / df["Close"].iloc[i]) > extrema):
            df.loc[i,"minmax"] = 1
        else:
            df.loc[i,"minmax"] = 0

    
    plt.figure(figsize=(16,8))
    plt.plot(df["Close"][4000:], color = 'black', label = 'Test')

    for index, row in df.iterrows():
        if row['minmax'] == 1:
            plt.axvline(x = index, color='red', linestyle='--', linewidth=0.8)
        if row['minmax'] == -1:
            plt.axvline(x = index, color='blue', linestyle='--', linewidth=0.8)
    plt.xlim(6000, df.index.max())

    plt.legend()
    plt.grid()

    st.header("Peaks and Troughs")
    st.write('''Let's first highlight some of the main, obvious trend changes in NVIDIA's stock chart. Below, the red lines represent 
             local highs and the blue represents local lows! To us, this graph looks like it is trending positively, but to accurately
             make predictions the model needs to know where exactly these trend changes occur!''')
    st.pyplot(plt)

    data['RSI']=ta.rsi(data.Close, length=15)
    data['EMAF']=ta.ema(data.Close, length=20)
    data['EMAM']=ta.ema(data.Close, length=100)
    data['EMAS']=ta.ema(data.Close, length=150)
    data['OBV']=ta.obv(data.Close, data.Volume, length=15)
    data['ATR']=ta.atr(data.High, data.Low, data.Close, length=15)

    data['Target'] = data['Close']-data.Open
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

    data['DeltaNextClose'] = (data['Close'].shift(-1))-data['Close']

    data.dropna(inplace=True)
    data.reset_index(inplace = True)
    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    data_set = data
    target_col = "TargetClass"
    from sklearn.preprocessing import MinMaxScaler
    sc_data = MinMaxScaler(feature_range=(0,1))
    sc_response = MinMaxScaler(feature_range=(0,1))

    response_scaled = sc_response.fit_transform(data_set[[target_col]])
    # print(response_scaled)
    # print(data_set.loc[:, data_set.columns != target_col])
    # print(response_scaled)
    data_set = data_set.drop(columns = ["DeltaNextClose", "Target"])
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
    X= []
    backcandles = 30

    for j in range(10):
        X.append([])
        for i in range(backcandles, data_set_scaled_new.shape[0]):
            X[j].append(data_set_scaled_new[i-backcandles:i,j])
    X = np.moveaxis(X,[0],[2])
    X, yi, = np.array(X), np.array(data_set_scaled_new[backcandles:,-1])
    y = np.reshape(yi,(len(yi),1))
    splitlimit = int(len(X)*0.8)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    # y_pred = model.predict(X_test)

    with open("pages/nvidia/nvidia2_y_pred.pkl", 'rb') as f:
        y_pred = pickle.load(f)

    plt.figure(figsize=(16,8))
    plt.plot(y_test, color = 'black', label = 'Test')
    plt.plot(y_pred, color = 'green', label = 'Predicted')
    plt.legend()
    plt.xlim(0, 225)

    st.header("Stock Movement Prediction")
    st.write("The model shows that for any given day it's guesses for whether a stock would move up or down is roughly 50%, which supports the efficient markets hypothesis!")
    st.pyplot(plt)



if __name__ == "__main__":
    main()



