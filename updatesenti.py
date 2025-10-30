import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model=pickle.load(open("sentu.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
review=st.text_input("Enter movie review: ")

if st.button("prediction"):
    review_scale=scaler.transform([review]).toarray()
    result=model.predict(review_scale)
    if result[0]==0:

        st.write("Negative Review")
    else:
        st.write("Positive Review")

