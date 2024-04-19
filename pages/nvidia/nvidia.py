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
        Page("pages/nvidia/david.py", "David", ":euro:"),
        Page("pages/nvidia/nvidia.py", "NVIDIA", ":pound:"),
        Page("pages/soon.py", "Coming Soon...", ":eyes:"),
    ]
)

st_pages.add_indentation()

def main():
    st.title("Shayan and David")
    # model = keras.models.load_model('pages/david/discreteStock.keras')
    data = yf.download(tickers = 'NVDA')
    df = data.copy()
    df.reset_index(inplace = True)
    extrema = 0.05
    for i in range(8,len(df)-8):
        if(((df["Adj Close"].iloc[i-7]-df["Adj Close"].iloc[i]) / df["Adj Close"].iloc[i]) > extrema) and (((df["Adj Close"].iloc[i+7]-df["Adj Close"].iloc[i]) / df["Adj Close"].iloc[i]) > extrema):
            df.loc[i,"minmax"] = -1
        elif (((df["Adj Close"].iloc[i]-df["Adj Close"].iloc[i+7]) / df["Adj Close"].iloc[i]) > extrema) and (((df["Adj Close"].iloc[i]-df["Adj Close"].iloc[i-7]) / df["Adj Close"].iloc[i]) > extrema):
            df.loc[i,"minmax"] = 1
        else:
            df.loc[i,"minmax"] = 0

    
    plt.figure(figsize=(16,8))
    plt.plot(df["Adj Close"][4000:], color = 'black', label = 'Test')

    for index, row in df.iterrows():
        if row['minmax'] == 1:
            plt.axvline(x = index, color='red', linestyle='--', linewidth=0.8)
        if row['minmax'] == -1:
            plt.axvline(x = index, color='blue', linestyle='--', linewidth=0.8)
    plt.xlim(6000, df.index.max())

    plt.legend()
    plt.grid()

    st.header("Graph of peaks and troughs")
    st.pyplot(plt)

    data['RSI']=ta.rsi(data.Close, length=15)
    data['EMAF']=ta.ema(data.Close, length=20)
    data['EMAM']=ta.ema(data.Close, length=100)
    data['EMAS']=ta.ema(data.Close, length=150)
    data['OBV']=ta.obv(data.Close, data.Volume, length=15)
    data['ATR']=ta.atr(data.High, data.Low, data.Close, length=15)

    data['Target'] = data['Adj Close']-data.Open
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

    data['DeltaNextClose'] = (data['Adj Close'].shift(-1))-data['Adj Close']

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
    plt.figure(figsize=(16,8))
    plt.plot(y_test, color = 'black', label = 'Test')
    # plt.plot(y_pred, color = 'green', label = 'pred')
    plt.legend()

    st.header("Graph of stock movement vs predicted")
    st.text("The model shows that for any given day it's guesses for whether a stock would move up or down is roughly 50%, which supports the efficient markets hypothesis")
    st.pyplot(plt)



if __name__ == "__main__":
    main()



