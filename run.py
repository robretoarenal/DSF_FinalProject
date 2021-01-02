import streamlit as st
from utils import NflPredict

if __name__ == "__main__":

    st.title('AJUAAA!')
    st.write('This app is the final project of DSF class')

    df=NflPredict().predict()

    #st.subheader('Selected values')
    st.header('NFL games week 17 are:')
    st.write(df)
