import streamlit as st
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
        Page("pages/nvidia/nvidia.py", "NVIDIA", ":euro:"),
        Page("pages/nvidia/nvidia2.py", "NVIDIA v2", ":pound:"),
        Page("pages/soon.py", "Coming Soon...", ":eyes:"),
    ]
)

st_pages.add_indentation()

cc1, cc2 = st.columns([0.8, 0.2])
with cc1:
    st.title("Stock Market Predictive Analysis")

with cc2:
    st.image("images/MDST_Logo.png", width=125)

st.write(
    '''
***

**⇦ Check the tabs to the left for research and model-specific results!** (top left corner for mobile users)

***

## Introduction

As daunting as the stock market may seem, it typically follows certain trends. Many of us have investments in the market, or will eventually, so it is important (and interesting) to understand what could influence stock prices. Using past market data, we can highlight the important factors that do just that. Then using that information, we can build a model to predict stock price trends!

This website contains the work done for the Stock Market Prediction project at the Michigan Data Science Team during the Winter semester of 2024. The project's main objective was to leverage machine learning methods to accurately predict individual stock trends and their primary factors, given a range of their historical data. Generating the next day's price was outside of the scope of the project.

## Stock Trend Prediction Models

Our approaches included using a LSTM neural network in conjunction with a variety of technical indicators from the `pandas_ta` library. Much of the focus of the project meetings were the integration of different indicators to specially hypertuned LSTMs. 

### 1. `pages/base/base_model.py -> Google`
Using a simple LSTM architecture, we trained a neural network on a long term Google Dataset. We transformed data from the `yfinance` API using `pandas_ta` to produce **momentum** based data elements. We leverages the **RSI** and **EMA** technical indicators to achieve this. The predictions of the model were generally poor, and could not generalize to volatility in stock price. This motivated our search for other indicators, datasets, and model architectures, to see if we could achieve better performance.

### 2. `pages/nvidia/nvidia.py -> NVIDIA`
We move to a more predicting a more volatile, recently popular stock. We used **Average True Range**  and **On-Balance Volume** in addition to our previous indicators to aid the volatility of the stock.

It turns out that the model performs quite well and it is generalizing well to values outside of the dataset since it uses trends as technical indicators to make inputs.

## Conclusions
The minimal availability of useful data along with the volatility of certain stocks make it abundantly clear why it is hard to predict stock trends. But we found alternative routes to make the project a positive learning experience, covering topics from interfacing with APIs in python, to researching the stock market for useful technical indicators and manually tuning hyperparameters of LSTM Neural Networks. Future iterations of the project should preprocess many stocks through a clustering to identify which stocks affect each other, and follow similar trends. Though this is a large undertaking, it adds a layer of practicality to the current model.

## Streamlit

We took advantage of the easy-to-use Streamlit library in Python to create a front-end to display the results of our research and models. The models are contained in the `streamlit/` directory. To run the app and see them, `pip install streamlit` and `python3 -m streamlit run home.py`. '''
)