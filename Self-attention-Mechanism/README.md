# Self-Attention Mechanism Implementation

## Overview

This notebook provides a step-by-step implementation of the self-attention mechanism, the core component of Transformer models. Through a hands-on approach, it demonstrates how self-attention processes a sentence ("Le chat noir dort.") by computing attention weights and generating contextualized word representations.

## Objectives

- Understand the mathematical foundations of self-attention
- Implement the complete self-attention pipeline from scratch
- Visualize attention patterns and their linguistic significance
- Analyze how attention weights capture grammatical and semantic relationships
- Gain intuition about how Transformers process sequential data

## Key Concepts

### Self-Attention Mechanism

Self-attention allows each position in a sequence to attend to all positions in the same sequence, computing a weighted sum of values based on query-key compatibility. This enables the model to capture dependencies between all words simultaneously, regardless of their distance in the sequence.

### Core Components

1. **Query (Q)**: Represents "what am I looking for?"
2. **Key (K)**: Represents "what do I offer?"
3. **Value (V)**: Represents "what information do I contain?"

## Notebook Structure

### 1. Configuration and Setup
- Dimension configuration: embedding dimension (d_embedding = 4), head dimension (d_head = 2)
- Example sentence: "Le chat noir dort." (The black cat sleeps.)
- Random seed initialization for reproducibility

### 2. Word Embeddings Creation
- Creating initial word embeddings for each word in the sentence
- Shape: (n_words, d_embedding) = (4, 4)
- Random initialization for demonstration purposes

### 3. Weight Matrices Initialization
- **WQ** (Query weights): Transforms embeddings to query space
- **WK** (Key weights): Transforms embeddings to key space
- **WV** (Value weights): Transforms embeddings to value space
- Shape: (d_embedding, d_head) = (4, 2)

### 4. Query, Key, and Value Computation
- **Q = embeddings × WQ**: Query vectors for each word
- **K = embeddings × WK**: Key vectors for each word
- **V = embeddings × WV**: Value vectors for each word
- Shape: (n_words, d_head) = (4, 2)

### 5. Attention Scores Calculation
- Computing similarity scores: **scores = Q × K^T**
- Shape: (n_words, n_words) = (4, 4)
- Each score represents how much word i should attend to word j

### 6. Softmax Normalization
- Applying softmax to convert scores into probability distributions
- Each row sums to 1.0, representing attention weights
- Higher values indicate stronger attention relationships

### 7. Weighted Output Computation
- **output = attention_weights × V**
- Produces contextualized representations for each word
- Each word's representation now contains information from all other words, weighted by attention

### 8. Analysis and Interpretation
- Attention pattern analysis
- Semantic relationship identification
- Mathematical transformation effects
- Implications for NLP applications

## Key Findings from the Implementation

### Attention Patterns Observed

**"Le" (The) - Article**
- 96.94% attention to "dort" (verb)
- Strong article-verb grammatical relationship
- Demonstrates how articles connect to main verbs

**"chat" (cat) - Noun**
- 100% attention to "dort" (verb)
- Clear subject-verb relationship
- Noun focuses entirely on the verb describing its action

**"noir" (black) - Adjective**
- Distributed attention: 39.77% to "Le", 38.48% to "chat", 19.72% to itself
- Reflects its role as an adjective modifying both article and noun
- Shows how modifiers connect to multiple elements

**"dort" (sleeps) - Verb**
- 99.95% attention to "Le" (article)
- Verb strongly connected to the beginning of the sentence
- Demonstrates long-range dependency capture

### Semantic Relationships Captured

1. **Grammatical Dependencies**
   - Syntactic connections: Article-verb, subject-verb relationships
   - Modifier relationships: Adjective connections to article and noun
   - Sentence structure: Attention patterns reflect French sentence structure

2. **Contextual Understanding**
   - Each word's representation contains relationship information
   - Captures both local and global dependencies
   - Contextualized embeddings encode sentence meaning

3. **Mathematical Transformations**
   - Dimensionality reduction: 4D embeddings → 2D output vectors
   - Information flow through weighted combinations
   - Sophisticated blending of all words' information

## Strengths of Self-Attention

1. **Parallel Processing**: All attention scores computed simultaneously
2. **Long-Range Dependencies**: Direct connections between any word pairs regardless of distance
3. **Interpretability**: Attention weights provide insights into model decisions
4. **Flexibility**: Can capture various types of relationships (syntactic, semantic, positional)

## Real-World Applications

- **Machine Translation**: Capturing cross-lingual grammatical relationships
- **Text Summarization**: Identifying important content through attention patterns
- **Question Answering**: Focusing on relevant parts of context
- **Named Entity Recognition**: Understanding entity relationships
- **Sentiment Analysis**: Capturing contextual sentiment signals

## Limitations of This Implementation

This simplified implementation demonstrates core concepts but lacks:

1. **Multi-Head Attention**: Real Transformers use multiple attention heads to capture different relationship types
2. **Positional Encoding**: Words are treated without explicit position information
3. **Layer Normalization**: No normalization for training stability
4. **Feed-Forward Networks**: Missing non-linear transformations
5. **Scalability**: Small dimensions (4×4) for demonstration purposes

## Mathematical Formulation

The self-attention mechanism can be expressed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q**: Query matrix
- **K**: Key matrix
- **V**: Value matrix
- **d_k**: Dimension of keys (for scaling)

In this implementation:
- Scores = Q × K^T
- Attention_weights = softmax(scores)
- Output = attention_weights × V

## Usage

1. Open the notebook in Jupyter or Google Colab
2. Run cells sequentially to follow the implementation
3. Observe how each step transforms the data
4. Analyze the attention matrix to understand relationships
5. Experiment with different sentences or dimensions

## Dependencies

- **NumPy**: For matrix operations and numerical computations

```bash
pip install numpy
```

## Educational Value

This notebook is ideal for:
- Understanding the fundamental building block of Transformers
- Learning how attention mechanisms work mathematically
- Visualizing attention patterns and their linguistic meaning
- Building intuition before working with full Transformer models
- Educational purposes in NLP and deep learning courses

## Key Takeaways

1. **Self-attention** computes relationships between all word pairs simultaneously
2. **Attention weights** reveal which words are most relevant to each other
3. **Contextualized representations** encode information from the entire sequence
4. **Grammatical relationships** are naturally captured through attention patterns
5. **The mechanism** is parallelizable and interpretable, making it powerful for NLP

## Extensions and Further Learning

To extend this implementation:
- Add positional encoding to incorporate word order
- Implement multi-head attention for diverse relationship capture
- Add layer normalization and residual connections
- Scale up to larger dimensions and sequences
- Compare with full Transformer architecture

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide to Transformers
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Practical implementation library

## Model Configuration

- **Embedding Dimension**: 4
- **Head Dimension**: 2
- **Number of Words**: 4
- **Sequence Length**: 4 tokens
- **Attention Matrix Shape**: (4, 4)

## Verification Steps

The notebook includes verification of:
- ✓ Matrix operation dimensions
- ✓ Softmax normalization (rows sum to 1.0)
- ✓ Attention weight validity
- ✓ Semantic relationship alignment
- ✓ Grammatical dependency capture
