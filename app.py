import streamlit as st
from rag_pipline import get_answer

question = st.text_input("Ask a question about the document")
# question = "What is semiconductor?"
if question:
    answer = get_answer(question)
    st.text_area("Answer", value=answer, height=200)
    # print(answer)