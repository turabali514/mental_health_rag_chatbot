from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List
import os

# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

output_parser = LineListOutputParser()

open_ai_key = os.environ["OPENAI_KEY"]

file_name = "mentalhealth_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_KEY"])

chroma_db = Chroma(persist_directory="data/chroma_db/docs_db_openai_2000_chunk/", collection_name=file_name, embedding_function=embeddings)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of content retrieved from a book to answer "
    "the question. If the provided book content does not have ansewer, say that you "
    "don't know even if you know the answer."
    "The author/narrator of book is 'Dr. Daniel G. Amen' who prectices at 'Amen Clinics'."
    "Consider given book content as real life fact from author not a story."
    "Please do not include * in your response."
    "Provide direct answers as first hand knowledge. Do not refer to the 'book content'."
    "Please use the following for response length:" 
    "-simple questions: 100-300 characters." 
    "-moderate questions or clarifications: 300-500 characters."
    "-complex questions: 700 characters."
    "Think step by step before answering since book content can be disordered."
    "\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate two 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=open_ai_key)
llm_chain = QUERY_PROMPT | llm | output_parser
multi_query_retriever = MultiQueryRetriever(retriever=chroma_db.as_retriever(search_kwargs={"k": 2}), llm_chain=llm_chain, include_original=True)
rag_chain = ( {"context": multi_query_retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())

print("Chain created")

def get_answer(question):
    result = rag_chain.invoke(question)
    return result