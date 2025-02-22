import streamlit as st
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
import base64

# Set Streamlit page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺", layout="centered")

# Load images for frontend
logo_image = Image.open("Static/cropped_image.webp")

# Assign the logo image as the bot avatar
bot_avatar = logo_image

# Load the user avatar image from the static directory
user_avatar = Image.open("Static/—Pngtree—user avatar placeholder white blue_6796231.png")



# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;  /* Dark background */
        color: #ffffff;  /* White text */
    }

    .stTextInput > div > div > input {
        background-color: #2e2e2e;  /* Dark input field */
        color: #ffffff;  /* White text */
        border: 1px solid #444;  /* Dark border */
    }

    .stButton > button {
        background-color: #4CAF50;  /* Green button */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }

    .stMarkdown h1 {
        color: #4CAF50;  /* Green title */
    }
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .chat-message img {
        margin-right: 10px;  /* Changed from left to right to place avatar on the left */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load API keys
try: 
    groq_api_key = st.secrets["GROQ_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    langchain_api_key = st.secrets["LANGSMITH-API-KEY"]
    hugging_face_api_key = st.secrets["HF_API_KEY"]
    langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]
except KeyError as e:
    st.error(f"Error: Missing API Key - {str(e)}")
    st.stop()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_PROJECT"]="Well MATE"  

# Validate API keys
if not pinecone_api_key:
    st.error("Error: PINECONE_API_KEY is missing!")
    st.stop()
if not groq_api_key:
    st.error("Error: GROQ_API_KEY is missing!")
    st.stop()

# Load Embeddings & Pinecone Vector Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_names = "healtcare-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_names,
    embedding=embeddings
)

# Display logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.image(logo_image, width=110)  # Adjust logo width as needed
with col2:
    st.markdown("<h1 style='text-align: left;'>WELLMATE ChatBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Your health matters—let’s heal together!</p>", unsafe_allow_html=True)

# Streamlit UI Setup
session_id="default_session"
# max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Set retriever
retriever = docsearch.as_retriever()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature= 0.7, max_tokens=200)

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
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# System prompt for answering questions
system_prompt = (
    "You are Dr. Black, a female physician specializing in general practice. Respond to healthcare queries with professionalism and empathy. "
    "Do not introduce yourself unless asked by the user. "
    "Provide clear, concise answers limited to five sentences. "
    "After your diagnosis, if applicable, prescribe medication by providing only the medicine name. in pdf form "
    "Utilize the retrieved context for your responses. If you lack the information to respond accurately, reply with 'I don't know'. "
    "Incorporate emojis where they can enhance communication or lighten the mood appropriately. \n\n {context}"
)
# Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),        
        ("human", "{input}"),
    ]
)

# Create retrieval and question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to manage session history
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

# Create conversational RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to render custom styled messages
def render_message(message, role):
    avatar = user_avatar if role == "user" else bot_avatar
    st.markdown(f'''
    <div class="chat-message">
        <img src="data:image/png;base64,{avatar_to_base64(avatar)}" width="30" height="30" />
        <div style="background-color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px 0; color: black;">
            {message}
        </div>
    </div>
    ''', unsafe_allow_html=True)

def avatar_to_base64(image):
    with open(image.filename, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Display chat history with custom avatars
for message in st.session_state.messages:
    render_message(message["content"], message["role"])

# User Input & Response Generation
user_input = st.chat_input("Your Message:")
if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id":session_id}
        },
    )
    bot_response = response['answer']
    
    # Append to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Display new messages
    render_message(user_input, "user")
    render_message(bot_response, "assistant")

