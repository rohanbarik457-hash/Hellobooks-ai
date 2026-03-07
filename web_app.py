import streamlit as st
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []

# CSS for Dark/Light mode overrides
if st.session_state.theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #000000; color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #111111; color: #ffffff; }
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        .stChatMessage { background-color: #1a1a1a !important; color: #ffffff !important; }
        .stChatMessage * { color: #ffffff !important; }
        h1, h2, h3, h4, h5, h6, p, span, label, div { color: #ffffff !important; }
        [data-testid="stChatInput"] textarea { background-color: #ffffff !important; color: #000000 !important; border: 1px solid #333333 !important; }
        [data-testid="stChatInput"] { background-color: #000000 !important; }
        [data-testid="stChatInput"] button { background-color: #000000 !important; color: #ffffff !important; }
        .stButton > button { background-color: #1a1a1a; color: #ffffff !important; border: 1px solid #333333; }
        .stButton > button:hover { background-color: #333333; }
        .stRadio label { color: #ffffff !important; }
        .stDownloadButton > button { background-color: #1a1a1a; color: #ffffff !important; border: 1px solid #333333; }
        .stMarkdown, .stCaption { color: #ffffff !important; }
        [data-testid="stHeader"] { background-color: #000000 !important; }
        [data-testid="stToolbar"] { background-color: #000000 !important; }
        [data-testid="stBottomBlockContainer"] { background-color: #000000 !important; }
        [data-testid="stBottom"] { background-color: #000000 !important; }
        [data-testid="stBottom"] > div { background-color: #000000 !important; }
        [data-testid="stChatInput"] > div { background-color: #000000 !important; }
        section[data-testid="stBottom"] { background-color: #000000 !important; }
        hr { border-color: #333333 !important; }
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
    
    if not st.session_state.messages:
        st.caption("No history yet.")
    else:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                time_str = msg.get("time", "")
                st.markdown(f"� **{time_str}**")
                st.markdown(f"  {msg['content'][:35]}...")
    
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.info("ℹ️ **Note on AI Limits**\nHellobooks uses Google Gemini (Free Tier).\nDaily Limit: 1,500 questions.\nIf you get a 'Limit Reached' error, please try again in 1 minute or the next day.")
    
    # Download current chat as text
    if st.session_state.messages:
        chat_text = ""
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Hellobooks AI"
            chat_text += f"{role}: {msg['content']}\n\n"
        st.download_button("📥 Download Chat", chat_text, file_name="hellobooks_chat.txt", mime="text/plain")
    
    # Saved chats section
    if st.session_state.saved_chats:
        st.markdown("---")
        st.title("💼 Saved Chats")
        for idx, saved in enumerate(st.session_state.saved_chats):
            label = f"{saved['title']}... ({saved['time']})"
            if st.button(label, key=f"load_{idx}"):
                st.session_state.messages = list(saved["messages"])
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
    # Add user question with real-time timestamp
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    st.session_state.messages.append({"role": "user", "content": question, "time": now})
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
