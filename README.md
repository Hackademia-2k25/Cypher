# Cypher
# Educational Assistant for Visually Impaired Learners

## Overview
This Educational Assistant is a specialized RAG (Retrieval-Augmented Generation) system designed to make learning resources accessible to visually impaired students. The system combines voice interaction, context-aware information retrieval, and natural language processing to provide an inclusive learning experience.

## Key Features
- **Voice-First Interaction**: Supports natural voice input and text-to-speech output
- **Context-Aware Responses**: Maintains conversation history to provide coherent, contextual explanations
- **Adaptive Learning**: Three primary modes of interaction:
  - Topic Explanation: Detailed explanations of concepts
  - Question Answering: Direct responses to specific queries
  - Interactive Practice: Voice-based Q&A sessions

## Technical Architecture
- **RAG System**: Combines retrieval from a knowledge base with generative AI responses
- **Vector Database**: Uses Chroma for efficient storage and retrieval of educational content
- **Language Model**: Powered by Groq's Mixtral-8x7b model for accurate and natural responses
- **Embeddings**: Utilizes HuggingFace's all-MiniLM-L6-v2 for semantic understanding

## Accessibility Features
- **Screen Reader Friendly**: All responses are optimized for text-to-speech conversion
- **Concise Responses**: Information is structured in easily digestible, three-sentence formats
- **Speaking Tone**: Natural, conversational responses for better engagement
- **Context Retention**: Remembers conversation history for more natural interactions

## Use Cases
- Self-paced learning for visually impaired students
- Interactive tutoring sessions
- Exam preparation and practice
- General knowledge exploration
