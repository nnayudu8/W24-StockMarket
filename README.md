# Stock Market Prediction - MDST Winter '24

Project Leads:
- Nidhil Nayudu (nnayudu@umich.edu)
- Jake Gwinn (jagwinn@umich.edu)

## Introduction

As daunting as the stock market may seem, it typically follows certain trends. Many of us have investments in the market, or will eventually, so it is important (and interesting) to understand what could influence stock prices. Using past market data, we can highlight the important factors that do just that. Then using that information, we can build a model to predict stock price trends!

This is a repository containing the work done for the Stock Market Prediction project at the Michigan Data Science Team during the Winter semester of 2024. The project's main objective was to leverage machine learning methods to accurately predict individual stock trends and their primary factors, given a range of their historical data. Generating the next day's price was outside of the scope of the project.

This repository contains several approaches to solving this task, using varied model architectures and datasets. We enumerate them below in the **Stock Trend Prediction Models** section. We also created front-end interfaces for displaying our results and getting predictions from our versatile, base model, which are described in the **Streamlit** section.

## Stock Trend Prediction Models

Our approaches included using a LSTM neural network in conjunction with a variety of technical indicators from the `pandas_ta` library. Much of the focus of the project meetings were the integration of different indicators to specially hypertuned LSTMs. 

### 1. `pages/base/base_model.py -> Google`
Using a simple LSTM architecture, we trained a neural network on a long term Google Dataset. We transformed data from the `yfinance` API using `pandas_ta` to produce **momentum** based data elements. We leverages the **RSI** and **EMA** technical indicators to achieve this. The predictions of the model were generally poor, and could not generalize to volatility in stock price. This motivated our search for other indicators, datasets, and model architectures, to see if we could achieve better performance.

### 2. `pages/nvidia/presentation.py -> NVIDIA`
We move to a more expansive dataset - a set of Amazon technology reviews. The LSTM had comparative performance, but had significantly better validation metrics and made more sensible predictions.

### 3. `models/GRU_amazon_reviews_model.ipynb`
Surprisingly, switching the LSTM layer to a GRU layer led to a boost in performance. The model jumped to 30% accuracy, and made the best predictions we had seen yet. After reducing the learning rate, we saw a rise to 50% accuracy.

### 4. `models/ngram_model.ipynb`
Using n-grams for predictions were never explicitly covered by the Fall '23 project, but is included in this repository for the sake of completeness. The Neural Net approaches all leveraged n-grams to generate input sequences that the models were trained on. But what if we cut out the neural net training process, and train the next word simply using next word probabilities derived from the n-grams themselves?

It turns out that the model performs quite well and it is generalizing well to values outside of the dataset since it uses trends as technical indicators to make inputs.

### Conclusions
The minimal availability of useful data along with the volatility of certain stocks make it abundantly clear why it is hard to predict stock trends. But we found alternative routes to make the project a positive learning experience, covering topics from interfacing with APIs in python, to researching the stock market for useful technical indicators and manually tuning hyperparameters of LSTM Neural Networks. Future iterations of the project should preprocess many stocks through a clustering to identify which stocks affect each other, and follow similar trends. Though this is a large undertaking, it adds a layer of practicality to the current model.

## Streamlit

We took advantage of the easy-to-use Streamlit library in Python to create a front-end to display the results of our research and models. The models are contained in the `streamlit/` directory. To run the app and see them, `pip install streamlit` and `python3 -m streamlit run home.py`. 

Alternatively, the deployed website can be viewed [here](https://mdst-next-word.streamlit.app/). This displays our general methods and results.