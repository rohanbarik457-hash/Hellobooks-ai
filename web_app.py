import streamlit as st
import os
import sys

# Inject Streamlit Cloud secrets into environment (for deployed app)
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.rag_pipeline import HellobooksRAG

st.set_page_config(page_title="Hellobooks AI", page_icon="📚", layout="centered", initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "messages" not in st.session_state:
    st.session_state.messages = []

# CSS for Dark/Light mode overrides
if st.session_state.theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #1E1E1E; color: #FFFFFF; }
        .stChatMessage { background-color: #2D2D2D !important; }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    try:
        return HellobooksRAG()
    except Exception as e:
        st.error(f"Error loading AI Assistant: {e}")
        return None

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Theme Toggle
    theme_choice = st.radio("UI Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()
        
    st.markdown("---")
    st.title("📜 Chat History")
    
    # Display mini history in sidebar
    if not st.session_state.messages:
        st.caption("No history yet.")
    else:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content'][:30]}...")
                
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main UI ---
st.title("📚 Hellobooks AI Assistant")
st.markdown("Ask me anything about bookkeeping, invoices, profit & loss, balance sheets, and cash flow!")

# Display full chat history in main window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

rag_system = load_rag_system()

if question := st.chat_input("Ask an accounting question..."):
    # Add user question to history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):
        if rag_system:
            with st.spinner("Thinking..."):
                answer = rag_system.answer_question(question)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("The AI Assistant is currently offline.")
