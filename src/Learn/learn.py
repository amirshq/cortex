from curses import meta
from json import load
from sys import implementation
from tkinter import N
from dotenv import load_dotenv
import os

from openai.types.shared import all_models, metadata

load_dotenv()
api_key = os.getenv()

#load config
#src/utils/config.py
def load_config(path:str = "src/config/config.yml") -> dict:
    with open(path, "r") as f:
        config = load(f)
    return config

#DTO
import openai
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
class ChatMessageRequest(BaseModel):
    message: str = Field(..., description="The message to be sent to the chatbot")
    user_id: Optional[int] = None
    session_id: Optional[str] = None
class ChatMessageResponse(BaseModel):
    response: str = Field(..., description="The response from the chatbot")
    session_id: Optional[str] = None
    timestamp: datetime

#Database Layer -> SQLite 
CREATE TABLE sessions (
    id TEXT primary key,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL DEFALU 'New conversation',
    created_at TEXT NOT NULL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
):


## Key Pattern in Python for SQLite management
import sqlite3
from pathlib import Path

class Chathistorymanager:
    def __init__(self, db_path:str):
        Path(db_path) = parents.mkdir(parents = True)

from abc import ABC, abstractmethod
class BaseLLM(ABC):
    def generate(self, question:str, contect:str="")-> str:
        """
        implementation of the LLM wrapper
        """
class OpenAIModel(BaseLLM):
    def __init__(self,model_name:str, temperature: float, max_tokens:int):
        from openai import OpenAI
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    def generate(self,question: str, context:str="")-> str: 
        messages = [{"role":"system","content":"You are a helpful assistant"},]
        if context:
            messages.append({"role":"user","content":f"Context\n{context}\n\nQuestion: {question}"})
        else:
            messages.append({"role":"user", "content":question})

        response = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
        )
        return response.choices[0].message.content

## system role
messages = [
    {"role":"system","content":"You are a helpful assistant"},
    {"role":"user","content":"what is the capital of france"},
    {"role":"assistant","content":"Paris"},
    {"role":"user","content":"And Germany?"},
]
reponse = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.7,
    max_tokens = 2500,
    messages = messages,
)
reply = reponse.choices[0].message.content
tokens_used = reponse.usage.total_tokens

# Tool Calling
tools = [
    {
        "type":"function",

    }

]


from openai import OpenAI
from typing import union 
class OpenAIEmbedder:
    def __init__(self, model_name:str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model_name = model_name 
    def embed_query(self,text:str )-> list[float]:
        """embed single string"""
        response = self.client.embeddings.create(
            model = self.model_name,
            input = text
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: list[str], batch_size: int= 50)-> list[list[float]]:
        all_embeddings = []
        for i in range (0,len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model = self.model_name,
                input = batch,
            )
            embeddings = [item.embedding for item in sored(response.data, )]
            all_embeddings.extend(embeddings)
        return all_embeddings


#Memory Layer (Vector DB) for POC level
import chromadb
from chromadb.config import Settings

class VectorDB:
    def __init__(self, persist_directory: str, collection_name: str):
        self.clinet = chromadb.PersistentClient(path= persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata = {"hnsw:space": "cosine"} )
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadata: list[dict],):
        self.collection.add(
            ids= ids,
            embeddings= embeddings,
            
        )



import chromadb
from chromadb.config import Settings

class VectorDB:
    def __init__(self, persist_directory:str, collection_name: str):
        self.clinet = chromadb.PersistentClient(path = persist_directory)
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        self.collection.add(
            ids = ids, 
            embeddings=embeddings, 
            documents = documents, 
            metadata = metadata, 
        )
    def search (self,query_embedding: list)