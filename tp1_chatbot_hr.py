"""
HR Chatbot - Natural Language Processing Project
================================================
A chatbot that answers HR-related questions using NLP techniques:
- Tokenization, Normalization, Lemmatization
- TF-IDF vectorization with bigrams
- Cosine similarity for response matching

Author: TP1 - NLP Course
Date: November 2025
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

import nltk
import random
import string
import re
import unicodedata
from collections import defaultdict
import warnings

# Download NLTK resources
print("Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

# Import sklearn for TF-IDF and similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
    sklearn_available = True
except ImportError:
    print("Sklearn not installed. Please install: pip install scikit-learn")
    sklearn_available = False

warnings.filterwarnings('ignore')
print("Libraries imported successfully!\n")


# ============================================================================
# STEP 2: Load HR Knowledge Base
# ============================================================================

print("=== Loading HR.txt ===")
try:
    # Try multiple encodings for compatibility
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    raw_data = None
    
    for encoding in encodings:
        try:
            with open('HR.txt', 'r', encoding=encoding, errors='ignore') as f:
                raw_data = f.read()
            print(f"HR.txt loaded successfully (encoding: {encoding})!")
            break
        except:
            continue
    
    if raw_data is None:
        print("Error: Unable to read HR.txt file.")
        exit()
    
    print(f"Total characters: {len(raw_data)}\n")
    
except FileNotFoundError:
    print("Error: HR.txt file not found.")
    exit()


# ============================================================================
# STEP 3: Convert to Lowercase
# ============================================================================

print("=== Converting to Lowercase ===")
raw_data = raw_data.lower()
print("Data converted to lowercase.\n")


# ============================================================================
# STEP 4: Display Last 1000 Characters
# ============================================================================

print("=== Last 1000 Characters ===")
print(raw_data[-1000:])
print("\n")


# ============================================================================
# STEP 5a: Tokenize Sentences
# ============================================================================

print("=== Sentence Tokenization ===")
sent_tokens = nltk.sent_tokenize(raw_data)
print(f"Number of sentences: {len(sent_tokens)}")
print(f"Example: {sent_tokens[0][:100]}...\n")


# ============================================================================
# STEP 5b: Text Normalization Function
# ============================================================================

print("=== Creating Normalization Function ===")

def normalize_text(text):
    """
    Normalize text by:
    - Removing accents (ASCII/UTF-8 decoding)
    - Removing punctuation
    - Removing digits
    - Removing extra whitespace
    
    Args:
        text (str): Input text to normalize
        
    Returns:
        str: Normalized text
    """
    # Remove accents and special characters
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Test normalization
test_sentence = sent_tokens[0]
print(f"Original: {test_sentence[:100]}...")
print(f"Normalized: {normalize_text(test_sentence)[:100]}...\n")


# ============================================================================
# STEP 6: Lemmatization
# ============================================================================

print("=== Lemmatization ===")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    """
    Lemmatize a list of tokens to their base form.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of lemmatized tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

# Apply normalization and lemmatization to all sentences
print("Applying normalization and lemmatization...")
normalized_sent_tokens = []

for sent in sent_tokens:
    normalized = normalize_text(sent)
    if normalized:  # Skip empty sentences
        word_tokens = nltk.word_tokenize(normalized)
        lemmatized = lemmatize_tokens(word_tokens)
        normalized_sent_tokens.append(' '.join(lemmatized))

print(f"Sentences after processing: {len(normalized_sent_tokens)}")
print(f"Example: {normalized_sent_tokens[0][:100]}...\n")


# ============================================================================
# BONUS STEPS: Chatbot Implementation
# ============================================================================

print("=== BONUS: Creating HR Chatbot ===\n")


# ============================================================================
# STEP 7: Welcome and Goodbye Dictionaries
# ============================================================================

print("Creating welcome and goodbye dictionaries...")

# Welcome patterns
welcome_inputs = [
    'bonjour', 'salut', 'hello', 'hi', 'hey', 'bonsoir', 
    'coucou', 'yo', 'allo', 'bonne journee'
]

welcome_responses = [
    "Hello! I'm an HR chatbot, how can I help you?",
    "Hi! I'm here to answer your HR questions.",
    "Welcome! Ask me anything about human resources.",
    "Hello! How can I assist you with HR matters today?",
    "Hi there! I'm your virtual HR assistant. How can I help?"
]

# Goodbye patterns
goodbye_inputs = [
    'au revoir', 'bye', 'salut', 'merci', 'a bientot', 
    'quit', 'exit', 'adieu', 'ciao', 'tchao'
]

goodbye_responses = [
    "Goodbye! Have a great day!",
    "See you later! Feel free to come back if you have more questions.",
    "Thank you for using the HR chatbot. Goodbye!",
    "Bye! Have an excellent day!",
    "Goodbye! Take care!"
]

print("Dictionaries created successfully!\n")


# ============================================================================
# STEP 8: Welcome Function
# ============================================================================

def welcome(user_input):
    """
    Check if user input is a greeting and return appropriate response.
    
    Args:
        user_input (str): User's input message
        
    Returns:
        str: Random welcome message if greeting detected, None otherwise
    """
    user_input_normalized = normalize_text(user_input.lower())
    
    for word in user_input_normalized.split():
        if word in welcome_inputs:
            return random.choice(welcome_responses)
    
    return None


# ============================================================================
# STEP 9: Response Generation Function
# ============================================================================

def generate_response(user_input):
    """
    Generate a response to user's question using TF-IDF and cosine similarity.
    
    Algorithm:
    1. Check for greetings
    2. Check for goodbye
    3. Vectorize input with TF-IDF (using bigrams)
    4. Calculate cosine similarity with all sentences
    5. Return most similar sentence if similarity > threshold
    
    Args:
        user_input (str): User's question
        
    Returns:
        str: Generated response or error message
    """
    if not sklearn_available:
        return "Sorry, response system unavailable. Please install scikit-learn."
    
    # Check for greeting
    welcome_resp = welcome(user_input)
    if welcome_resp:
        return welcome_resp
    
    # Check for goodbye
    user_input_normalized = normalize_text(user_input.lower())
    for word in user_input_normalized.split():
        if word in goodbye_inputs:
            return random.choice(goodbye_responses)
    
    # Add user's question to sentence corpus
    # Use original sentences (lowercase) for better context preservation
    sent_tokens_copy = sent_tokens.copy()
    sent_tokens_copy.append(user_input.lower())
    
    try:
        # Create TF-IDF matrix with optimized parameters
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,            # Consider all words
            stop_words=None      # Don't filter stop words for HR context
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(sent_tokens_copy)
        
        # Calculate cosine similarity between query and all sentences
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        # Find most similar sentence
        similar_index = cosine_similarities.argsort()[0][-1]
        similarity_score = cosine_similarities[0][similar_index]
        
        # Reduced threshold for more flexible matching
        if similarity_score < 0.05:
            return "Sorry, I couldn't find relevant information on this topic. Could you rephrase your question?"
        
        # Return the most similar original sentence
        return sent_tokens[similar_index]
    
    except Exception as e:
        return "Sorry, I couldn't process your question. Could you rephrase it?"


# ============================================================================
# STEP 10: Interactive Chatbot Interface
# ============================================================================

def start_chatbot():
    """
    Start the interactive chatbot interface.
    Allows user to ask questions and receive responses until they quit.
    """
    print("\n" + "="*70)
    print("       HR CHATBOT - HUMAN RESOURCES ASSISTANT       ")
    print("="*70)
    print("\nHello! I'm an HR chatbot, how can I help you?")
    print("\n[INFO] Tips:")
    print("  - Ask me questions about human resources")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*70 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        # Check for empty input
        if not user_input:
            print("Chatbot: Please enter a question.\n")
            continue
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'au revoir', 'bye']:
            print(f"Chatbot: {random.choice(goodbye_responses)}\n")
            break
        
        # Generate and display response
        response = generate_response(user_input)
        print(f"Chatbot: {response}\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    if sklearn_available:
        print("\n[OK] All steps completed successfully!")
        print("\n[START] Launching chatbot...\n")
        start_chatbot()
    else:
        print("\n[WARNING] Please install scikit-learn to use the chatbot:")
        print("    pip install scikit-learn")
        print("\n[OK] Steps 1-6 completed successfully!")
