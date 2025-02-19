import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# import sentence_transformers
import os

#fetching API from secret.toml file
 
try: 
    groq_api_key = st.secrets["GROQ_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    langchain_api_key = st.secrets["LANGSMITH-API-KEY"]
    hugging_face_api_key = st.secrets["HF_API_KEY"]
    langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]
except KeyError as e:
    st.error(f"Error: Missing API Key - {str(e)}")
    st.stop()
# Retrieve the value from st.secrets
# Set the environment variable
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"]="Well MATE "  

# Validate API keys

if not pinecone_api_key:
    st.error("Error: PINECONE_API_KEY is missing!")
    st.stop()
if not groq_api_key:
    st.error("Error: GROQ_API_KEY is missing!")
    st.stop()

# Load Embeddings & Pinecone Vector Store

embeddings =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_names = "healtcare-chatbot" 

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_names,
    embedding=embeddings
)


st.title("WELL MATE - Healthcare Chatbot")

# Streamlit UI Setup
session_id=st.sidebar.text_input("Session ID",value="default_session")
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

#set retriever
retriever = docsearch.as_retriever()

# Initialize LLM
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile",temperature=temperature,max_tokens=max_tokens)

# Ensure session state is initialized
if 'store' not in st.session_state:
    st.session_state["store"] = {}



# System prompt for question contextualization

contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

# Create history-aware retriever
history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

# System prompt for answering questions

system_prompt = (
    
    "You are a female Doctor named Black, assisting with healthcare-related queries. "
    "Do not introduce yourself unless the user asks. "
    "Provide clear and concise answers in a maximum of five sentences. "
    "Use the provided retrieved context for answering. If you don't know, say 'I don't know'. "
    "Use emojis when appropriate. \n\n {context}"

) 

# Chat prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        MessagesPlaceholder("chat_history"),        
        ("human", "{input}"),
    ]
)

# Create retrieval and question-answering chain

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain= create_retrieval_chain(history_aware_retriever,question_answer_chain)

# Function to manage session history
def get_session_history(session:str)->BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id]=ChatMessageHistory()
    return st.session_state["store"][session_id]

# Create conversational RAG chain

conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
)



# User Input & Response Generation

user_input = st.text_input("Your question:")
if user_input:
    session_history=get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id":session_id}
        },  # constructs a key "abc123" in `store`.
    )

    # Display chatbot response

    # st.write(st.session_state.store)
    st.write("Assistant:", response['answer'])
    # st.write("Chat History:", session_history.messages)
