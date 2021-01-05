import streamlit as st
from utils import ScoresPredict

if __name__ == "__main__":

    st.title('NFL games predictions')
    st.write('This model predicts the winning probabilities for each team.')
    #st.write('This app is the final project of DSF class')

    #image = Image.open('nfl-1.jpg')
    #st.image(image, use_column_width=True)
    if st.button('Predict'):
        df=ScoresPredict().predict()
        st.header('Games of the week are:')
        st.table(df.assign(hack='').set_index('hack'))
