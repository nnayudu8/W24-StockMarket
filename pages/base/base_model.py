import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import yfinance as yf
import pandas_ta as ta
import pickle

import st_pages
from st_pages import Page, show_pages, Section, add_page_title

st.set_page_config(layout = "wide")

# user_input = st.text_input("Input a stock ticker", "GOOG")

# tick = yf.Ticker(user_input)

# company_name = tick.info['longName']

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

data = pd.read_csv("pages/base/base_files/df.csv")
# Explaination 

cc5, cc6 = st.columns([0.8, 0.2])
with cc5:
    st.title("Initial Model")

with cc6:
    st.image("images/google_logo.jpg", width=100)

st.markdown("""
            This is the initial model that was used as a baseline for learning tensorflow and getting aquainted with technical indicators
            """)



st.header("Initial Plots", divider="gray")
cc1, cc2 = st.columns([2.5,5])
with cc1:

    st.subheader("The Data")
    st.dataframe(data, height=250, width = 1000)


with cc2:
    st.subheader(f"Closing Prices of Google")
    st.line_chart(data=data["Adj Close"])

st.header("Feature Engineering", divider="gray")
st.markdown("""
            """)

# data['RSI']=ta.rsi(data.Close, length=15)
# data['EMAF']=ta.ema(data.Close, length=20)
# data['EMAM']=ta.ema(data.Close, length=100)
# data['EMAS']=ta.ema(data.Close, length=150)

# data['Target'] = data['Adj Close']-data.Open
# data['Target'] = data['Target'].shift(-1)

# data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

# data['TargetNextClose'] = data['Adj Close'].shift(-1)

# data.dropna(inplace=True)
# data.reset_index(inplace = True)

# data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
# data_set = data
target_col = "TargetNextClose"

data_set = pd.read_csv("pages/base/base_files/data_set.csv")

cc3, cc4 = st.columns([4.5,5])
with cc3:
    st.code("""# Technical Indicators 
data['RSI']=ta.rsi(data.Close, length=15)
data['EMAF']=ta.ema(data.Close, length=20)
data['EMAM']=ta.ema(data.Close, length=100)
data['EMAS']=ta.ema(data.Close, length=150)

    # Different Response Variables
    data['Target'] = data['Adj Close']-data.Open
    data['Target'] = data['Target'].shift(-1)
    data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
    data['TargetNextClose'] = data['Adj Close'].shift(-1)

    data.dropna(inplace=True)
    data.reset_index(inplace = True)
    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    data_set = data
    
    target_col = "TargetNextClose"
        """)

with cc4:
    st.dataframe(data)
    
from sklearn.preprocessing import MinMaxScaler
sc_data = MinMaxScaler(feature_range=(0,1))
sc_response = MinMaxScaler(feature_range=(0,1))

response_scaled = sc_response.fit_transform(data_set[[target_col]])

data_set_scaled = sc_data.fit_transform(data_set.loc[:, data_set.columns != target_col])

data_set_scaled_new = []
for i in range(len(data_set_scaled)):
    data_set_scaled_new.append([])

    data_set_scaled_new[i] = np.append(data_set_scaled[i],response_scaled[i])


data_set_scaled_new = np.asarray(data_set_scaled_new)

st.header("Data Scaling", divider="gray")
st.markdown("""
            We then needed to scale the data in order to have the best performance out of our model. We scaled the model to be between 0 and 1.
            """)

st.code("""
from sklearn.preprocessing import MinMaxScaler
sc_data = MinMaxScaler(feature_range=(0,1))
sc_response = MinMaxScaler(feature_range=(0,1))

response_scaled = sc_response.fit_transform(data_set[[target_col]])

data_set_scaled = sc_data.fit_transform(data_set.loc[:, data_set.columns != target_col])

data_set_scaled_new = []
for i in range(len(data_set_scaled)):
    data_set_scaled_new.append([])

    data_set_scaled_new[i] = np.append(data_set_scaled[i],response_scaled[i])

data_set_scaled_new = np.asarray(data_set_scaled_new)
""")



X = []


backcandles = 30

for j in range(8):
    X.append([])
    for i in range(backcandles, data_set_scaled_new.shape[0]):#backcandles+2
        X[j].append(data_set_scaled_new[i-backcandles:i, j])


X=np.moveaxis(X, [0], [2])
X, yi =np.array(X), np.array(data_set_scaled_new[backcandles:,-1])

y=np.reshape(yi,(len(yi),1))


st.header("Formatting Data", divider="gray")
st.markdown("""
            Since we want to predict future days using past data. We need to format the data in a way where we are feeding in a certian number of past days worth of data into the model
            """)
st.code("""
X = []

backcandles = 30 # Using 30 days worth of data to predict next day

for j in range(8):#data_set_scaled_new[0].size):#2 columns are target not X
    X.append([])
    for i in range(backcandles, data_set_scaled_new.shape[0]):#backcandles+2
        X[j].append(data_set_scaled_new[i-backcandles:i, j])

#move axis from 0 to position 2
X=np.moveaxis(X, [0], [2])

# Choose -1 for last column, classification else -2...
X, yi =np.array(X), np.array(data_set_scaled_new[backcandles:,-1])
y=np.reshape(yi,(len(yi),1))

""")


splitlimit = int(len(X)*0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

st.header("Splitting the Data", divider="gray")
st.code("""
splitlimit = int(len(X)*0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
        """)


# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import Dense
# from keras.layers import TimeDistributed

# import tensorflow as tf
# import keras
# from keras import optimizers
# from keras.callbacks import History
# from keras.models import Model
# from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
# import numpy as np
# #tf.random.set_seed(20)
# np.random.seed(10)

# lstm_input = Input(shape=(backcandles, 8), name='lstm_input')
# inputs = LSTM(150, name='first_layer')(lstm_input)
# inputs = Dense(1, name='dense_layer')(inputs)
# output = Activation('linear', name='output')(inputs)
# model = Model(inputs=lstm_input, outputs=output)
# adam = optimizers.Adam()
# model.compile(optimizer=adam, loss='mse')
# model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=False, validation_split = 0.1)


st.header("Running the Model", divider="gray")
st.code("""
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
#tf.random.set_seed(20)
np.random.seed(10)

lstm_input = Input(shape=(backcandles, 8), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=False, validation_split = 0.1)
""")

st.header("Results", divider="gray")
y_pred = pd.read_csv("pages/base/base_files/y_pred.csv" )
y_pred.drop(y_pred.columns[0], axis=1, inplace=True)
y_pred = y_pred.to_numpy()
# y_pred = y_pred.to_list()

# st.write(y_pred)
# st.write(type(y_pred))

# Y_pred = []

# # for i in len(y_pred):


# st.dataframe(y_pred, use_container_width=True)
# st.dataframe(y_test, use_container_width=True)


# final = pd.DataFrame(
#     {
#         'Predicted': y_pred,
#         'Test': y_test
#     }
# )

# st.line_chart( data = final)

st.image("images/base_result.png")

st.write("Evidently, this simple, base model that we used for learning is extremely poor at predicting volatile stocks. We will prioritize that in our model creation going forward!")