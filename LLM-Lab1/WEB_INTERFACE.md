# 🌐 Web Interface Guide

## 🚀 Launch the Web Interface

```bash
streamlit run app.py
```

The interface will automatically open in your browser at: **http://localhost:8501**

If it doesn't open automatically, manually navigate to: `http://localhost:8501`

---

## ✨ Interface Features

### 💬 Chat Interface
- **Modern Design**: Clean and professional UI
- **Real-time Responses**: Instant chatbot replies
- **Color-coded Messages**: 
  - 🔵 Blue boxes = Your messages
  - 🟢 Green boxes = Bot responses

### 📊 Confidence Scores
Every bot response shows:
- **🎯 High confidence** (score > 20%)
- **✅ Good match** (score > 0%)
- **🤔 Low confidence** (score near 0%)

### 🎯 Quick Start Buttons
Click any button to instantly ask:
- "What is HR analytics?"
- "Tell me about recruitment"
- "What are employee rights?"
- "Tell me about compensation"

### 📝 Sidebar Features
- **ℹ️ About**: Project information
- **💡 Tips**: How to ask better questions
- **📝 Example Questions**: Inspiration for queries
- **🗑️ Clear History**: Start fresh conversation
- **ℹ️ Statistics**: View message count

---

## 📱 How to Use

### Step 1: Start a Conversation
Type your question in the chat input at the bottom:
```
Type your question here... (e.g., 'What is HR analytics?')
```

### Step 2: Get Instant Responses
The bot will:
1. Process your question using NLP
2. Search the HR knowledge base
3. Return the most relevant answer
4. Show confidence score

### Step 3: Continue the Conversation
- Ask follow-up questions
- Try different topics
- Use the quick start buttons
- Clear history to start fresh

---

## 💡 Tips for Best Results

### ✅ Good Questions
- "What is HR analytics?"
- "Tell me about employee rights"
- "What is recruitment?"
- "Tell me about compensation"
- "What is HR planning?"

### ❌ Avoid
- Too vague: "Tell me something"
- Off-topic: "What's the weather?"
- No keywords: "Give me information"

### 🎯 Pro Tips
1. **Use HR keywords**: analytics, recruitment, compensation, rights
2. **Be specific**: Ask about one topic at a time
3. **English works best**: Better results than French
4. **Check confidence**: Higher scores = better matches

---

## 🎨 Interface Preview

```
┌─────────────────────────────────────────────────────────────┐
│  💼 HR Chatbot Assistant                         [Sidebar] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  👋 Hello! I'm your HR assistant. Ask me anything!         │
│                                                              │
│  🚀 Quick Start Questions                                   │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ 📊 HR analytics? │  │ ⚖️ Employee      │               │
│  └──────────────────┘  │    rights?       │               │
│  ┌──────────────────┐  └──────────────────┘               │
│  │ 👥 Recruitment?  │  ┌──────────────────┐               │
│  └──────────────────┘  │ 💰 Compensation? │               │
│                        └──────────────────┘               │
│                                                              │
│  ─────────────────────────────────────────────────────      │
│                                                              │
│  [Chat messages appear here]                                │
│                                                              │
│  👤 You:                                                    │
│  What is HR analytics?                                      │
│                                                              │
│  🤖 HR Assistant 🎯:                                        │
│  HR analytics is about metrics and measurement...          │
│  Confidence: 85%                                            │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  Type your question here... [Send]                          │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Technical Details

### Built With
- **Streamlit**: Web framework
- **NLTK**: Natural language processing
- **Scikit-learn**: Machine learning (TF-IDF)
- **Python 3.10**: Programming language

### How It Works
1. **User Input**: You type a question
2. **Preprocessing**: Text normalization and tokenization
3. **TF-IDF**: Convert text to numerical vectors
4. **Similarity**: Calculate cosine similarity
5. **Response**: Return most similar sentence
6. **Display**: Show with confidence score

### Performance
- **Response Time**: < 1 second
- **Accuracy**: 65% response rate
- **Knowledge Base**: 177 sentences
- **Threshold**: 5% similarity minimum

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill existing Streamlit process
Ctrl + C in terminal

# Or use a different port
streamlit run app.py --server.port 8502
```

### Interface Not Loading
1. Check terminal for errors
2. Ensure HR.txt is in the same folder
3. Verify all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

### No Responses / Errors
1. **Check HR.txt**: File must exist
2. **Restart app**: Ctrl+C then rerun
3. **Clear cache**: Settings → Clear cache in Streamlit

---

## 🎓 Keyboard Shortcuts

- **Enter**: Send message
- **Ctrl + C**: Stop the server
- **Ctrl + R**: Refresh the page
- **Esc**: Clear input field

---

## 📊 Statistics & Monitoring

The interface tracks:
- **Total messages**: Count of all messages
- **Confidence scores**: Average accuracy
- **Response times**: Speed of replies

View statistics by clicking "Show Statistics" in sidebar.

---

## 🚀 Deployment Options

### Local (Current)
```bash
streamlit run app.py
```

### Share with Others
```bash
# Make accessible on your network
streamlit run app.py --server.address 0.0.0.0
```

### Deploy Online (Future)
- **Streamlit Cloud**: Free hosting
- **Heroku**: Cloud platform
- **AWS/Azure**: Enterprise hosting

---

## 📝 Feedback & Support

- **Issues**: Check terminal output
- **Questions**: See README.md
- **Improvements**: Suggest features

---

## ✨ Enjoy Your HR Chatbot!

The web interface is now running at: **http://localhost:8501**

Start asking questions and exploring HR topics! 🚀

---

**Built with ❤️ using Streamlit, NLTK, and Scikit-learn**

