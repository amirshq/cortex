# Memory Management

This module implements short-term and long-term memory management for the chatbot.

## Short-Term Memory (Redis)

Stores recent interactions (minutes to hours):
- Conversation context
- Last N conversation turns
- **Storage**: Exact text

## Long-Term Memory (Database)

Stores persistent data (days, weeks, months, or permanent):
- User profiles
- Historical chats
- Factual knowledge
- **Storage**: Semantic meaning

## Key Difference

- **Short-term** = Exact text storage
- **Long-term** = Semantic meaning storage
