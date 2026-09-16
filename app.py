import streamlit as st
import re
import joblib
def clean_txt(text):
    x=re.sub(r"<.*?>|[.,_,\",-,!,\',:,),(,-]"," ",text)
    x=x.lower()
    return x
model=joblib.load('sentiment_model.pkl')
vectorizer=joblib.load('tfidf_vectorizer.pkl')
st.title('تحلیل احساس نظرات فیلم')
user_input = st.text_area('نظرت رو اینجا بنویس:')
if st.button('تحلیل کن'):
    cleaned=clean_txt(user_input)
    vec=vectorizer.transform([cleaned])
    prediction=model.predict(vec)
    labels = {0: "Bad", 1: "Good"}
    result = labels[prediction[0]]
    st.write(result)