import streamlit as st
import joblib
from pipeline_utils import clean_text

@st.cache_resource
def load_model():
    return joblib.load('artifacts/sentiment_model.joblib')

st.set_page_config(page_title='Customer Review Sentiment',page_icon='💬',layout='centered')
st.title('💬 Customer Review Sentiment Analyzer')
st.write('Enter a review to classify its sentiment as positive or negative.')
model=load_model()
review=st.text_area('Customer review',placeholder='Write a customer review here...')
if st.button('Analyze sentiment',type='primary'):
    if not review.strip(): st.warning('Please enter a review.')
    else:
        cleaned=clean_text(review)
        pred=model.predict([cleaned])[0]
        st.subheader('Prediction')
        st.success(pred) if pred=='Positive' else st.error(pred)
        if hasattr(model,'predict_proba'):
            probs=model.predict_proba([cleaned])[0]
            st.caption('Model confidence estimates')
            for label,value in zip(model.classes_,probs):
                st.write(f'{label}: {value:.1%}')
                st.progress(float(value))
        with st.expander('Cleaned text'):
            st.write(cleaned)
