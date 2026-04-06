# Only src/ is deployed
# Tests are not included
# No pytest installed
```

---

## Real Example: Your RAG Chatbot

**Project structure:**
```
agentic-rag-mvp/
├── src/
│   ├── __init__.py
│   ├── chatbot.py          ← What the client runs
│   ├── memory.py
│   ├── retriever.py
│   └── orchestrator.py
├── tests/
│   ├── test_chatbot.py     ← You run locally
│   ├── test_memory.py      ← CI/CD runs before deploy
│   └── test_retriever.py
├── requirements.txt        ← Production dependencies only
├── requirements-dev.txt    ← pytest, black, mypy, etc.
└── .github/workflows/test.yml  ← Runs tests on every push
```

**requirements.txt (production):**
```
anthropic==0.45.0
azure-ai-foundry==0.1.0
langgraph==0.2.0
chromadb==0.5.0
```

**requirements-dev.txt (development only):**
```
-r requirements.txt
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.0
black==23.0.0
mypy==1.0.0