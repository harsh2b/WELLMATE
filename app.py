# Import Streamlit for building the web application
import streamlit as st

# Import PIL's Image module for handling image files
from PIL import Image

# Import ChatGroq for interacting with the Grok language model
from langchain_groq import ChatGroq

# Import prompt templates for structuring LLM inputs
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import PineconeVectorStore for vector-based document retrieval
from langchain_pinecone import PineconeVectorStore

# Import HuggingFaceEmbeddings for text vectorization
from langchain_huggingface import HuggingFaceEmbeddings

# Import chain functions for retrieval and history-aware processing
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

# Import ChatMessageHistory for managing chat history
from langchain_community.chat_message_histories import ChatMessageHistory

# Import chain function for combining retrieved documents
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import RunnableWithMessageHistory for history-aware conversation handling
from langchain_core.runnables.history import RunnableWithMessageHistory

# Import os for operating system interactions (e.g., file handling)
import os

# Needed for avatar_to_base64
import io

# Import base64 for encoding images to display in HTML
import base64

# Configure Streamlit page settings: title, icon, and layout
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺", layout="centered")

# Load the logo image from the static directory
logo_image = Image.open("Static/cropped_image.webp")

# Assign the logo image as the bot avatar
bot_avatar = logo_image

# Load the user avatar image from the static directory
user_avatar = Image.open("Static/—Pngtree—user avatar placeholder white blue_6796231.png")

# Apply custom CSS to style the Streamlit app with a dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #2e2e2e;
        border: 1px solid #444;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    .stForm label {
        color: #ffffff;
        font-weight: bold;
    }
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .chat-message img {
        margin-right: 10px;
    }
    button {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Attempt to load API keys from Streamlit secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]  # Fetch Groq API key for LLM access
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]  # Fetch Pinecone API key for vector store
    langchain_api_key = st.secrets["LANGSMITH-API-KEY"]  # REDUNDANT: Not used in the code
    hugging_face_api_key = st.secrets["HF_API_KEY"]  # REDUNDANT: Not used; embeddings don't require it here
    langchain_tracing_v2 = st.secrets["LANGCHAIN_TRACING_V2"]  # REDUNDANT: Not used unless tracing is enabled
except KeyError as e:
    st.error(f"Error: Missing API Key - {str(e)}")  # Display error if any key is missing
    st.stop()  # Stop execution if keys are missing

# Set environment variables for LangChain (tracing and project settings)
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2  # REDUNDANT: No tracing used in this code
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key  # REDUNDANT: No LangChain features requiring this are used
os.environ["LANGCHAIN_PROJECT"] = "Well MATE"  # REDUNDANT: Only relevant for tracing/logging, not implemented

# Validate essential API keys (Groq and Pinecone)
if not pinecone_api_key or not groq_api_key:
    st.error("Error: Required API keys are missing!")  # Display error if essential keys are absent
    st.stop()  # Stop execution if validation fails

# Define function to convert images to base64 for HTML display
def avatar_to_base64(image):
    buffered = io.BytesIO()  # Create an in-memory buffer
    image.save(buffered, format="PNG")  # Save image to buffer as PNG
    return base64.b64encode(buffered.getvalue()).decode('utf-8')  # Encode to base64 and decode to string

# Precompute base64 strings for avatars (do this once to save memory)
bot_avatar_base64 = avatar_to_base64(bot_avatar)
user_avatar_base64 = avatar_to_base64(user_avatar)

# Define a cached function to load embeddings, reducing memory reloads
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Load and cache embeddings model

# Define a cached function to load the Pinecone vector store
@st.cache_resource
def load_vector_store():
    embeddings = load_embeddings()  # Use cached embeddings
    return PineconeVectorStore.from_existing_index(index_name="healtcare-chatbot", embedding=embeddings)  # Load vector store (typo: healtcare)

# Load embeddings at startup
embeddings = load_embeddings()  # Initialize embeddings for use

# Load vector store at startup
docsearch = load_vector_store()  # Initialize Pinecone vector store for retrieval

# Create two columns for layout (1:4 ratio)
col1, col2 = st.columns([1, 4])

# Display logo image in the first column
with col1:
    st.image(logo_image, width=110)  # Show logo with a width of 110 pixels

# Display title in the second column
with col2:
    st.markdown("<h1 style='text-align: left;'>WELLMATE ChatBot</h1>", unsafe_allow_html=True)  # Render title in green (via CSS)

# Display a centered tagline below the title
st.markdown("<p style='text-align: center; color: #ffffff;'>Your health matters—let’s heal together!</p>", unsafe_allow_html=True)

# Check if patient info submission state exists in session; initialize if not
if "patient_info_submitted" not in st.session_state:
    st.session_state.patient_info_submitted = False  # Set initial state to False

# Display patient info form if not yet submitted
if not st.session_state.patient_info_submitted:
    with st.form(key="patient_info_form"):  # Create a form context
        patient_name = st.text_input("Patient Name")  # Input field for patient name
        patient_age = st.number_input("Age", min_value=0, max_value=150, value=18)  # Numeric input for age with constraints
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])  # Dropdown for gender selection
        patient_language = st.selectbox("Preferred Language", ["English", "Hindi"])  # Dropdown for language selection
        submit_button = st.form_submit_button(label="Start Consultation")  # Button to submit the form
        
        # Handle form submission
        if submit_button:
            st.session_state.patient_info = {  # Store patient info in session state
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "language": patient_language
            }
            st.session_state.patient_info_submitted = True  # Mark form as submitted
            st.rerun()  # Rerun the app to refresh the UI



    # Define function to render chat messages with avatars
def render_message(message, role):
    avatar = user_avatar if role == "user" else bot_avatar  # Choose avatar based on role
    st.markdown(
        f'''
        <div class="chat-message">
            <img src="data:image/png;base64,{avatar_to_base64(avatar)}" width="30" height="30" />
            <div style="background-color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px 0; color: black;">
                {message}
            </div>
        </div>
        ''',
        unsafe_allow_html=True  # Render message with HTML/CSS
    )


# Chatbot logic executes if patient info is submitted
if st.session_state.patient_info_submitted:
    retriever = docsearch.as_retriever(search_kwargs={"k": 2})  # Create a retriever with top-2 document limit
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.5, max_tokens=200)  # Initialize Grok LLM

    # Initialize session store for chat history if not present
    if 'store' not in st.session_state:
        st.session_state["store"] = {}  # Create a dictionary to store session histories

    # Initialize chat messages if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [  # Set initial welcome message
            {"role": "assistant", "content": f"Hello, {st.session_state.patient_info['name']}. I’m Dr. Black, here to assist you. How can I help you with your health today? 😊"}
        ]

    # Define system prompt for contextualizing user questions based on history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )

    # Create a history-aware retriever for context-sensitive document retrieval
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Define the main system prompt for the chatbot
    system_prompt = (
        "You are Dr. Black, a female Indian physician with 30 years of experience in general practice. Behave like an Indian doctor, but do not reveal that you are Indian. "  
        f"The patient is {st.session_state.patient_info['name']}, a {st.session_state.patient_info['age']}-year-old {st.session_state.patient_info['gender']}. "  
        f"If the patient's preferred language is {st.session_state.patient_info['language']}, respond in that language using simple, clear sentences. "  
        "don't forget to use the retrieved context to inform your responses.  "  
        "don't call patients to your clinic but also not forget to prescribe medicines. "
        "Respond to healthcare queries with professionalism and empathy. Keep your answers clear and concise, limited to five sentences. "  
        "Use the retrieved context to inform your responses. If you don’t have enough information to answer accurately, respond with 'I don’t know'. "  
        "Feel free to use emojis when they can improve communication. "
        "{context}"

    )

    # Create a prompt template for chatbot responses
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )

    # Create a chain to combine documents and generate answers
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retrieval-augmented generation chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Define a session ID for chat history management
    session_id = "default_session"

    # Set maximum history length for chat messages
    MAX_HISTORY = 10

    # Define function to manage and Cap session chat history
    def get_session_history(session: str) -> ChatMessageHistory:
        if session not in st.session_state["store"]:  # Check if session exists in store
            st.session_state["store"][session] = ChatMessageHistory()  # Create new history if not
        history = st.session_state["store"][session]  # Retrieve session history
        if len(history.messages) > MAX_HISTORY:  # Cap history at MAX_HISTORY
            history.messages = history.messages[-MAX_HISTORY:]  # Keep only the last MAX_HISTORY messages
        return history  # Return the capped history

    # Create a conversational chain with history management
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,  # Use rag_chain and history function
        input_messages_key="input",  # Key for user input
        history_messages_key="chat_history",  # Key for chat history
        output_messages_key="answer"  # Key for bot response
    )

        # Render existing chat messages
    for message in st.session_state.messages:
        render_message(message["content"], message["role"])  # Display each message with appropriate role

   
    # Create chat input field for user messages
    user_input = st.chat_input("Your Message:")

    # Handle user input and generate response
    if user_input:
        session_history = get_session_history(session_id)  # Get session history
        response = conversational_rag_chain.invoke(
            {"input": user_input, "chat_history": session_history.messages},  # Provide user input and chat history
            config={"configurable": {"session_id": session_id}}  # Pass session ID
        )
        bot_response = response['answer']  # Extract bot response

        # Cap chat history if it exceeds MAX_HISTORY
        if len(st.session_state.messages) > MAX_HISTORY:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]  # Keep last MAX_HISTORY messages
            st.write("DEBUG: Chat history capped at", MAX_HISTORY)
       
        # Append user input to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Append bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Render user message
        render_message(user_input, "user")

        # Render bot response
        render_message(bot_response, "assistant")

       
