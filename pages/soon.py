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
        Page("pages/nvidia/david.py", "David", ":euro:"),
        Page("pages/nvidia/nvidia.py", "NVIDIA", ":pound:"),
        Page("pages/soon.py", "Coming Soon...", ":eyes:"),
    ]
)

st_pages.add_indentation()

st.title("Coming Soon...")

st.write(
    '''
    At the moment, not everyone's front end page is complete, but come back soon to see our full results! 👀
'''
)