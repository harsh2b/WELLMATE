import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


#fetching API from secret.toml file 
groq_api_key = st.secrets["GROQ_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
langchain_api_key = st.secrets["LANGSMITH-API-KEY"]

# Retrieve the value from st.secrets
langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]

# Set the environment variable
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"]="Well MATE "


## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("ai","You are a female helpful assistant Named Dr. Black. Please  repsonse to the user queries"),
        ("human","Question:{question}")
    ]
)


def generate_response(question,llm,temperature,max_tokens):
    
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


st.title("WELL MATE")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")
## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Go  ahead and ask any question")
user_input=st.text_input("You:")

if user_input :
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")

