"""
HR Chatbot - Web Interface
===========================
Modern web interface for the HR chatbot using Streamlit.

Run with: streamlit run app.py
"""

import streamlit as st
import nltk
import random
import string
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="HR Chatbot",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #1976D2;
    }
    .user-message .message-header {
        color: #0D47A1;
    }
    .user-message .message-content {
        color: #1565C0;
        font-size: 1rem;
    }
    .bot-message {
        background-color: #F1F8E9;
        border-left: 5px solid #388E3C;
    }
    .bot-message .message-header {
        color: #1B5E20;
    }
    .bot-message .message-content {
        color: #2E7D32;
        font-size: 1rem;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .message-content {
        line-height: 1.6;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Initialize Session State
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False


# ============================================================================
# Load Chatbot Resources
# ============================================================================

@st.cache_resource
def initialize_chatbot():
    """
    Initialize chatbot resources (loaded once and cached).
    Downloads NLTK data and loads HR knowledge base.
    """
    try:
        # Download NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Load HR.txt
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        raw_data = None
        
        for encoding in encodings:
            try:
                with open('HR.txt', 'r', encoding=encoding, errors='ignore') as f:
                    raw_data = f.read()
                break
            except:
                continue
        
        if raw_data is None:
            return None, None, None
        
        # Process data
        raw_data = raw_data.lower()
        sent_tokens = nltk.sent_tokenize(raw_data)
        
        # Define welcome/goodbye patterns
        welcome_inputs = ['bonjour', 'salut', 'hello', 'hi', 'hey', 'bonsoir']
        goodbye_inputs = ['au revoir', 'bye', 'quit', 'exit', 'aurevoir']
        
        return sent_tokens, welcome_inputs, goodbye_inputs
        
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None, None, None


# ============================================================================
# Text Processing Functions
# ============================================================================

def normalize_text(text):
    """Normalize text by removing punctuation, digits, and extra spaces."""
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_greeting(user_input, welcome_inputs):
    """Check if user input is a greeting."""
    user_input_normalized = normalize_text(user_input.lower())
    for word in user_input_normalized.split():
        if word in welcome_inputs:
            return True
    return False


def is_goodbye(user_input, goodbye_inputs):
    """Check if user input is a goodbye."""
    user_input_normalized = normalize_text(user_input.lower())
    for word in user_input_normalized.split():
        if word in goodbye_inputs:
            return True
    return False


# ============================================================================
# Response Generation
# ============================================================================

def generate_response(user_input, sent_tokens, welcome_inputs, goodbye_inputs):
    """
    Generate response using TF-IDF and cosine similarity.
    
    Args:
        user_input: User's question
        sent_tokens: List of sentences from knowledge base
        welcome_inputs: List of greeting patterns
        goodbye_inputs: List of goodbye patterns
        
    Returns:
        tuple: (response, similarity_score)
    """
    # Check for greeting
    if is_greeting(user_input, welcome_inputs):
        responses = [
            "Hello! 👋 I'm an HR chatbot. How can I help you today?",
            "Hi there! 😊 Ask me anything about human resources!",
            "Welcome! I'm here to answer your HR questions."
        ]
        return random.choice(responses), 1.0
    
    # Check for goodbye
    if is_goodbye(user_input, goodbye_inputs):
        responses = [
            "Goodbye! 👋 Have a great day!",
            "See you later! Feel free to come back anytime. 😊",
            "Thank you for using the HR chatbot. Goodbye! 👋"
        ]
        return random.choice(responses), 1.0
    
    # Generate response with TF-IDF
    sent_tokens_copy = sent_tokens.copy()
    sent_tokens_copy.append(user_input.lower())
    
    try:
        # TF-IDF with bigrams
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(sent_tokens_copy)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        similar_index = cosine_similarities.argsort()[0][-1]
        similarity_score = cosine_similarities[0][similar_index]
        
        # Check threshold
        if similarity_score < 0.05:
            return "Sorry, I couldn't find relevant information on this topic. Could you rephrase your question? 🤔", 0.0
        
        return sent_tokens[similar_index], similarity_score
        
    except Exception as e:
        return f"Sorry, I encountered an error processing your question. 😕", 0.0


# ============================================================================
# Main Interface
# ============================================================================

# Header
st.title("💼 HR Chatbot Assistant")
st.markdown("### Ask me anything about Human Resources!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses **Natural Language Processing** to answer HR questions:
    
    - 📚 **Knowledge Base**: HR.txt
    - 🤖 **Algorithm**: TF-IDF + Cosine Similarity
    - 🎯 **Accuracy**: 65% response rate
    
    ---
    
    ### 💡 Tips
    - Ask specific questions
    - Use HR-related keywords
    - English works best
    
    ### 📝 Example Questions
    - "What is HR analytics?"
    - "Tell me about employee rights"
    - "What is recruitment?"
    - "Tell me about compensation"
    
    ---
    
    ### 🔄 Actions
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("ℹ️ Show Statistics"):
        st.info(f"Total messages: {len(st.session_state.messages)}")
    
    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("- Python 🐍")
    st.markdown("- NLTK 📖")
    st.markdown("- Scikit-learn 🤖")
    st.markdown("- Streamlit 🎈")

# Initialize chatbot
if not st.session_state.chatbot_initialized:
    with st.spinner("🚀 Loading HR knowledge base..."):
        sent_tokens, welcome_inputs, goodbye_inputs = initialize_chatbot()
        
        if sent_tokens is None:
            st.error("❌ Failed to load HR.txt. Please ensure the file exists.")
            st.stop()
        
        st.session_state.sent_tokens = sent_tokens
        st.session_state.welcome_inputs = welcome_inputs
        st.session_state.goodbye_inputs = goodbye_inputs
        st.session_state.chatbot_initialized = True
        st.success(f"✅ Loaded {len(sent_tokens)} sentences from knowledge base!")

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">👤 You</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        score = message.get("score", 0)
        score_emoji = "🎯" if score > 0.2 else "✅" if score > 0 else "🤔"
        
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-header">🤖 HR Assistant {score_emoji}</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if score > 0 and score < 1.0:
            st.caption(f"✨ Confidence: {score:.2%}")

# Chat input
user_input = st.chat_input("Type your question here... (e.g., 'What is HR analytics?')")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.spinner("🤔 Thinking..."):
        response, score = generate_response(
            user_input,
            st.session_state.sent_tokens,
            st.session_state.welcome_inputs,
            st.session_state.goodbye_inputs
        )
    
    # Add bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "score": score
    })
    
    # Rerun to display new messages
    st.rerun()

# Welcome message if no messages yet
if len(st.session_state.messages) == 0:
    st.info("👋 Hello! I'm your HR assistant. Ask me anything about human resources!")
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Start Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 What is HR analytics?"):
            question = "What is HR analytics?"
            st.session_state.messages.append({"role": "user", "content": question})
            # Generate response immediately
            response, score = generate_response(
                question,
                st.session_state.sent_tokens,
                st.session_state.welcome_inputs,
                st.session_state.goodbye_inputs
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "score": score
            })
            st.rerun()
        
        if st.button("👥 Tell me about recruitment"):
            question = "Tell me about recruitment"
            st.session_state.messages.append({"role": "user", "content": question})
            # Generate response immediately
            response, score = generate_response(
                question,
                st.session_state.sent_tokens,
                st.session_state.welcome_inputs,
                st.session_state.goodbye_inputs
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "score": score
            })
            st.rerun()
    
    with col2:
        if st.button("⚖️ What are employee rights?"):
            question = "What are employee rights?"
            st.session_state.messages.append({"role": "user", "content": question})
            # Generate response immediately
            response, score = generate_response(
                question,
                st.session_state.sent_tokens,
                st.session_state.welcome_inputs,
                st.session_state.goodbye_inputs
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "score": score
            })
            st.rerun()
        
        if st.button("💰 Tell me about compensation"):
            question = "Tell me about compensation"
            st.session_state.messages.append({"role": "user", "content": question})
            # Generate response immediately
            response, score = generate_response(
                question,
                st.session_state.sent_tokens,
                st.session_state.welcome_inputs,
                st.session_state.goodbye_inputs
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "score": score
            })
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>💼 HR Chatbot | Built with NLP & Machine Learning</p>
    <p style='font-size: 0.8rem;'>Powered by NLTK, Scikit-learn & Streamlit</p>
</div>
""", unsafe_allow_html=True)

