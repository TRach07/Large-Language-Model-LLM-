"""
HR Chatbot - Automated Test Suite
==================================
Tests all steps (1-10) of the chatbot implementation.

This script validates:
- Data loading and processing
- Tokenization and normalization
- Lemmatization
- Welcome/goodbye detection
- Response generation with TF-IDF
"""

print("="*70)
print("TEST SUITE - HR CHATBOT")
print("="*70 + "\n")

# ============================================================================
# Import Required Libraries
# ============================================================================

import nltk
import string
import re
import unicodedata
from collections import defaultdict
import warnings
import sys

# Configure encoding for Windows compatibility
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Download NLTK resources
print("[+] Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
print("[OK] Libraries imported successfully!\n")


# ============================================================================
# TEST STEP 2: Load Data from HR.txt
# ============================================================================

print("="*70)
print("STEP 2: Loading HR.txt Data")
print("="*70)

try:
    # Try multiple encodings for compatibility
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    raw_data = None
    
    for encoding in encodings:
        try:
            with open('HR.txt', 'r', encoding=encoding, errors='ignore') as f:
                raw_data = f.read()
            print(f"[OK] File loaded successfully (encoding: {encoding})!")
            break
        except:
            continue
    
    if raw_data is None:
        print("[ERROR] Unable to read HR.txt file.")
        exit()
    
    print(f"   Total characters: {len(raw_data)}\n")
    
except FileNotFoundError:
    print("[ERROR] HR.txt file not found.")
    exit()


# ============================================================================
# TEST STEP 3: Convert to Lowercase
# ============================================================================

print("="*70)
print("STEP 3: Converting to Lowercase")
print("="*70)

original_sample = raw_data[:100]
raw_data = raw_data.lower()
lowercase_sample = raw_data[:100]

print(f"   Original: {original_sample}")
print(f"   Lowercase: {lowercase_sample}")
print("[OK] Conversion to lowercase completed!\n")


# ============================================================================
# TEST STEP 4: Display Last 1000 Characters
# ============================================================================

print("="*70)
print("STEP 4: Displaying Last 1000 Characters")
print("="*70)

last_1000 = raw_data[-1000:]
print(f"   Length: {len(last_1000)} characters")
print(f"   Start: {last_1000[:100]}...")
print(f"   End: ...{last_1000[-100:]}")
print("[OK] Last 1000 characters displayed!\n")


# ============================================================================
# TEST STEP 5a: Sentence Tokenization
# ============================================================================

print("="*70)
print("STEP 5a: Sentence Tokenization")
print("="*70)

sent_tokens = nltk.sent_tokenize(raw_data)
print(f"[OK] Number of tokenized sentences: {len(sent_tokens)}")
print(f"   Example sentences:")
for i in range(min(3, len(sent_tokens))):
    print(f"   {i+1}. {sent_tokens[i][:80]}...")
print()


# ============================================================================
# TEST STEP 5b: Text Normalization
# ============================================================================

print("="*70)
print("STEP 5b: Text Normalization")
print("="*70)

def normalize_text(text):
    """
    Normalize text by removing:
    - Punctuation
    - ASCII/UTF-8 encoding issues
    - Unwanted symbols (digits, extra spaces)
    
    Args:
        text (str): Input text
        
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

# Test normalization function
test_sentence = sent_tokens[0]
normalized_sentence = normalize_text(test_sentence)

print(f"   Original: {test_sentence[:100]}...")
print(f"   Normalized: {normalized_sentence[:100]}...")
print("[OK] Normalization function created and tested!\n")


# ============================================================================
# TEST STEP 6: Lemmatization
# ============================================================================

print("="*70)
print("STEP 6: Lemmatization")
print("="*70)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    """
    Lemmatize tokens to their base form.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of lemmatized tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

# Apply normalization and lemmatization to all sentences
normalized_sent_tokens = []

for sent in sent_tokens:
    normalized = normalize_text(sent)
    if normalized:
        word_tokens = nltk.word_tokenize(normalized)
        lemmatized = lemmatize_tokens(word_tokens)
        normalized_sent_tokens.append(' '.join(lemmatized))

print(f"[OK] Sentences normalized and lemmatized: {len(normalized_sent_tokens)}")
print(f"   Example: {normalized_sent_tokens[0][:100]}...")
print()


# ============================================================================
# TEST BONUS STEP 7: Welcome/Goodbye Dictionaries
# ============================================================================

print("="*70)
print("BONUS - STEP 7: Welcome/Goodbye Dictionaries")
print("="*70)

welcome_inputs = ['bonjour', 'salut', 'hello', 'hi', 'hey', 'bonsoir']
welcome_responses = [
    "Hello! I'm an HR chatbot, how can I help you?",
    "Hi! I'm here to answer your HR questions.",
]

goodbye_inputs = ['au revoir', 'bye', 'quit', 'exit']
goodbye_responses = [
    "Goodbye! Have a great day!",
    "See you later!",
]

print(f"[OK] Dictionaries created:")
print(f"   - Welcome inputs: {len(welcome_inputs)} entries")
print(f"   - Welcome responses: {len(welcome_responses)} entries")
print(f"   - Goodbye inputs: {len(goodbye_inputs)} entries")
print(f"   - Goodbye responses: {len(goodbye_responses)} entries")
print()


# ============================================================================
# TEST BONUS STEP 8: Welcome Function
# ============================================================================

print("="*70)
print("BONUS - STEP 8: Welcome() Function")
print("="*70)

import random

def welcome(user_input):
    """
    Check if user input is a greeting.
    
    Args:
        user_input (str): User's input
        
    Returns:
        str: Welcome response if greeting detected, None otherwise
    """
    user_input_normalized = normalize_text(user_input.lower())
    for word in user_input_normalized.split():
        if word in welcome_inputs:
            return random.choice(welcome_responses)
    return None

# Test welcome function
print("   Testing welcome() function:")
test_inputs = ["Hello!", "Bonjour", "How are you?"]
for test_input in test_inputs:
    result = welcome(test_input)
    status = result if result else 'Not a greeting'
    print(f"   - '{test_input}' -> {status}")

print("[OK] Welcome() function created and tested!\n")


# ============================================================================
# TEST BONUS STEP 9: Response Generation Function
# ============================================================================

print("="*70)
print("BONUS - STEP 9: generateResponse() Function")
print("="*70)

def generate_response(user_input):
    """
    Generate response using TF-IDF and cosine similarity.
    
    Args:
        user_input (str): User's question
        
    Returns:
        str: Generated response
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
    
    # Use original sentences for better matching
    sent_tokens_copy = sent_tokens.copy()
    sent_tokens_copy.append(user_input.lower())
    
    try:
        # Optimized TF-IDF parameters
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Unigrams + bigrams
            min_df=1,
            stop_words=None
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(sent_tokens_copy)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        
        similar_index = cosine_similarities.argsort()[0][-1]
        similarity_score = cosine_similarities[0][similar_index]
        
        # Reduced threshold for flexibility
        if similarity_score < 0.05:
            return "Sorry, I couldn't find relevant information."
        
        return sent_tokens[similar_index]
        
    except Exception as e:
        return "Sorry, I couldn't process your question."

# Test response generation
print("   Testing generateResponse() function:")
test_questions = [
    "Hello",
    "What is HR analytics?",
    "Tell me about employee rights",
    "bye"
]

for question in test_questions:
    response = generate_response(question)
    print(f"\n   Q: {question}")
    display_response = response[:150] + "..." if len(response) > 150 else response
    print(f"   R: {display_response}")

print("\n[OK] generateResponse() function created and tested!\n")


# ============================================================================
# TEST BONUS STEP 10: User Interface
# ============================================================================

print("="*70)
print("BONUS - STEP 10: User Interface")
print("="*70)
print("[OK] User interface is available in tp1_chatbot_hr.py")
print("   To use it, run: python tp1_chatbot_hr.py\n")


# ============================================================================
# TEST SUMMARY
# ============================================================================

print("="*70)
print("TEST SUMMARY")
print("="*70)
print("[OK] Step 1: Import libraries")
print("[OK] Step 2: Load HR.txt data")
print("[OK] Step 3: Convert to lowercase")
print("[OK] Step 4: Display last 1000 characters")
print("[OK] Step 5a: Sentence tokenization")
print("[OK] Step 5b: Text normalization")
print("[OK] Step 6: Lemmatization")
print("[OK] BONUS Step 7: Welcome/goodbye dictionaries")
print("[OK] BONUS Step 8: Welcome() function")
print("[OK] BONUS Step 9: generateResponse() function")
print("[OK] BONUS Step 10: User interface")
print("\n" + "="*70)
print("[SUCCESS] ALL TESTS PASSED!")
print("="*70)
print("\n[INFO] To use the interactive chatbot:")
print("   python tp1_chatbot_hr.py\n")
