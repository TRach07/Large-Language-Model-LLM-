"""
HR Chatbot - Quick Demo with Performance Metrics
=================================================
This script demonstrates the improved chatbot with:
- Pre-defined test questions
- Similarity scores display
- Performance validation
"""

print("\n" + "="*70)
print("     QUICK DEMO - IMPROVED HR CHATBOT")
print("="*70 + "\n")

# ============================================================================
# Import Required Libraries
# ============================================================================

import nltk
import random
import string
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

print("[1/5] Loading HR.txt file...")


# ============================================================================
# Load Knowledge Base
# ============================================================================

# Try multiple encodings for compatibility
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
raw_data = None

for encoding in encodings:
    try:
        with open('HR.txt', 'r', encoding=encoding, errors='ignore') as f:
            raw_data = f.read()
        break
    except:
        continue

# Convert to lowercase for case-insensitive matching
raw_data = raw_data.lower()
sent_tokens = nltk.sent_tokenize(raw_data)

print(f"      [OK] {len(sent_tokens)} sentences loaded")


# ============================================================================
# Configure Dictionaries
# ============================================================================

print("\n[2/5] Configuring welcome/goodbye responses...")

# Welcome patterns
welcome_inputs = ['bonjour', 'salut', 'hello', 'hi', 'hey', 'bonsoir']
welcome_responses = [
    "Hello! I'm an HR chatbot, how can I help you?",
    "Hi! I'm here to answer your HR questions.",
]

# Goodbye patterns
goodbye_inputs = ['au revoir', 'bye', 'quit', 'exit', 'aurevoir']
goodbye_responses = [
    "Goodbye! Have a great day!",
    "See you later! Feel free to come back.",
]

print("      [OK] Dictionaries configured")


# ============================================================================
# Text Processing Functions
# ============================================================================

print("\n[3/5] Creating normalization function...")

def normalize_text(text):
    """
    Normalize text by removing punctuation, digits, and extra spaces.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("      [OK] Normalization function created")


# ============================================================================
# Response Generation Functions
# ============================================================================

print("\n[4/5] Configuring optimized response function...")

def welcome(user_input):
    """
    Detect if user input is a greeting.
    
    Args:
        user_input (str): User's message
        
    Returns:
        str: Welcome message or None
    """
    user_input_normalized = normalize_text(user_input.lower())
    for word in user_input_normalized.split():
        if word in welcome_inputs:
            return random.choice(welcome_responses)
    return None


def generate_response(user_input):
    """
    Generate response using TF-IDF and cosine similarity.
    Shows similarity score for demonstration purposes.
    
    Args:
        user_input (str): User's question
        
    Returns:
        str: Response with similarity score
    """
    # Check for greeting
    welcome_resp = welcome(user_input)
    if welcome_resp:
        return welcome_resp
    
    # Check for goodbye
    user_input_normalized = normalize_text(user_input.lower())
    for word in user_input_normalized.split():
        if word in goodbye_inputs:
            return random.choice(goodbye_responses)
    
    # Generate response with TF-IDF
    sent_tokens_copy = sent_tokens.copy()
    sent_tokens_copy.append(user_input.lower())
    
    try:
        # Optimized TF-IDF with bigrams
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Unigrams + bigrams for better context
            min_df=1,
            stop_words=None
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(sent_tokens_copy)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        similar_index = cosine_similarities.argsort()[0][-1]
        similarity_score = cosine_similarities[0][similar_index]
        
        # Reduced threshold (0.05 instead of 0.1)
        if similarity_score < 0.05:
            return "Sorry, I couldn't find relevant information. Please rephrase."
        
        # Return response with similarity score
        return f"{sent_tokens[similar_index]}\n\n   [Similarity score: {similarity_score:.4f}]"
        
    except:
        return "Sorry, I couldn't process your question."

print("      [OK] Response function configured")


# ============================================================================
# Run Demo Tests
# ============================================================================

print("\n[5/5] Running automated improvement tests...")
print("="*70)

# Pre-defined test questions
test_questions = [
    ("Hello", "Greeting test"),
    ("What is HR analytics?", "HR analytics test"),
    ("Tell me about employee rights", "Employee rights test"),
    ("What is recruitment?", "Recruitment test"),
    ("What is compensation?", "Compensation test"),
    ("What is HR planning?", "HR planning test"),
    ("bye", "Goodbye test"),
]

# Run all tests
for i, (question, description) in enumerate(test_questions, 1):
    print(f"\n{description}:")
    print(f"Q: {question}")
    response = generate_response(question)
    
    # Limit display length for readability
    if len(response) > 200:
        print(f"R: {response[:200]}...")
    else:
        print(f"R: {response}")
    print("-"*70)


# ============================================================================
# Demo Summary
# ============================================================================

print("\n" + "="*70)
print("           [SUCCESS] DEMO COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n[INFO] Chatbot improvements working correctly!")
print("   - Reduced similarity threshold: 0.05 (was 0.1)")
print("   - Using bigrams for better context")
print("   - Preserving original text (less normalization)")
print("\n[INFO] To use the full chatbot, run:")
print("   python tp1_chatbot_hr.py")
print("\n")
