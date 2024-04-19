import streamlit as st

# Define the layout of the home page
def home():
    st.title("Home Page")
    st.write('This is our first model: A model of the Block Stock using')
    st.write("Welcome to the Home Page!")
    st.write("Please select a page from the sidebar to view images.")

# Define the layout of the first page
def page1():
    st.title("Page 1")
    st.write("This is our first model: A model of the Block Stock using DEMA and EMA")
    uploaded_file1 = st.file_uploader("Upload Screenshot 1", type="PNG")
    if uploaded_file1 is not None:
        st.image(uploaded_file1, caption="Uploaded Screenshot 1")
    
    st.write("Here is the accuracy of our model:")
    
    uploaded_file2 = st.file_uploader("Upload Screenshot 2", type="PNG")
    if uploaded_file2 is not None:
        st.image(uploaded_file2, caption="Uploaded Screenshot 2")
    
    st.write("We learned that EMA and DEMA work well to make a good model for a volatile stock")

# Define the layout of the second page
def page2():
    st.title("Page 2")
    st.write("This is our first model: A model of the Block Stock using RSI and Bias")
    
    uploaded_file1 = st.file_uploader("Upload Screenshot 1", type=["png", "jpg", "jpeg"])
    if uploaded_file1 is not None:
        st.image(uploaded_file1, caption="Uploaded Screenshot 1")
    
    st.write("Here is the accuracy of our model")
    
    uploaded_file2 = st.file_uploader("Upload Screenshot 2", type="PNG")
    if uploaded_file2 is not None:
        st.image(uploaded_file2, caption="Uploaded Screenshot 2")
    
    st.write("We learned that RSI and Bias work well to make a good model for a volatile stock")

# Main function to run the app
def main():
    page = st.sidebar.selectbox("Select Page", ["Home", "Page 1", "Page 2"])
    if page == "Home":
        home()
    elif page == "Page 1":
        page1()
    elif page == "Page 2":
        page2()

if __name__ == "__main__":
    main()



