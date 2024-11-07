import streamlit as st
from rag_pipline import get_answer

question = st.text_input("Ask a question")
if question:
    answer = get_answer(question)
    st.text_area("Answer", value=answer, height=400)