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
        Page("pages/nvidia/david.py", "David", ":euro:"),
        Page("pages/nvidia/nvidia.py", "NVIDIA", ":pound:"),
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
    So in this graph, there is a 10, 21, and 50 day EMA, aka fast, medium, and slow moving averages, respectively. As the name
    indicates, the fast moving average adjusts to recent day much more precisely, whereas the slow moving average captures more
    long term trends. So to interpret this, if a stock's price crosses multiple EMAs, a new trend is generally indicated. So
    for example, the first circle indicates when the start of a positive trend can be assumed!
    ***
    '''
    )

st.subheader("Relative Strength Index (RSI)")
