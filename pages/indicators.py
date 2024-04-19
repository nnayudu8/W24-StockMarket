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

st.title("Technical Indicators")

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

st.write('''
         ***
         **Below you can find an explanation of some of the technical indicators that we used during our project!**
         ***
         ''')

st.subheader("Exponential Moving Averages (EMAs)")
st.write(
    '''
    EMAs are perfect for trend prediction, and are heavily used in day trading. Essentially, you specify a range of days for 
    the EMA to account for. It determines an average trend during that range.
    '''
    )
st.image("images/EMA_graph.png")
st.write(
    '''
    So in this graph, there are 10, 21, and 50 day EMAs, aka fast, medium, and slow moving averages, respectively. As the name
    indicates, the fast moving average adjusts to recent day much more precisely, whereas the slow moving average captures more
    long term trends. So to interpret this, if a stock's price crosses multiple EMAs, a new trend is generally indicated. So
    for example, the first circle indicates when the start of a positive trend can be assumed!
    ***
    '''
    )

st.subheader("Relative Strength Index (RSI)")
st.write(
    '''
    The basic idea of gaming the stock market is to buy low and sell high. RSI is a great indicator to do just that!
    '''
    )
st.image("images/RSI.png")
st.write(
    '''
    The values of RSI range between 0 and 100. A stock is considered oversold when RSI is closer to 0, and overbought when it 
    is closer to 100. This is calculated using the gains and losses over a time interval, so a generally accepted pattern is 
    to buy when below 30 and sell when over 70!
    ***
    '''
    )

st.subheader("Average Directional Index (ADX)")
st.write(
    '''
    Above, we saw EMAs are great for determining trends. But trends can be weak or strong, how do we know to trust it with our 
    money? Well, that's primarily what ADX is used for!
    '''
    )
st.image("images/ADX.png")
st.write(
    '''
    The values of ADX range between 0 and 100. Note that ADX **IS NOT** a bullish or bearish signal, it just determines the 
    strength of a trend. When ADX is below 25, it is considered a weak trend, and when above, it is considering strong. So ideally, 
    we want a positive moving EMA with as high of a corresponding ADX value as possible!
    ***
    '''
    )
