import hashlib
import json 
from typing import Dict, Optional
from redis.asyncio import Redis

DEFAULT_TTL = 900 # 15 minutes

"""
SIMPLE EXPLANATION: What is ResponseCache?

Think of it like a "smart notebook" that remembers answers to questions:

1. User asks: "What's the weather?"
2. System checks notebook: "Do I have an answer for this?"
3. If YES → Return saved answer (FAST! No need to ask LLM & No need to consume tokens)
4. If NO → Ask LLM, save answer in notebook, return answer

WHY IT'S USEFUL:
- Saves money (don't call expensive LLM for same question)
- Faster responses (instant if cached)
- Reduces load on LLM servers

HOW IT WORKS (CORRECT FLOW):
1. Convert QUESTION to hash (creates unique fingerprint of the question)
2. Create a key using: user_id + model + question_hash
3. Check if answer exists in cache using this key
4. If exists → Return the saved answer (stored as JSON, NOT hashed)
5. If not exists → Ask LLM, store answer (as JSON), return answer

IMPORTANT: We hash the QUESTION, NOT the answer!
- Question → Hashed → Used as part of the key
- Answer → Stored as JSON → Retrieved when key matches
"""

class ResponseCache:
    """
    A Redis-based cache for storing and retrieving LLM responses.
    
    Simple analogy: Like a smart notebook that remembers answers to questions
    so you don't have to ask the same question twice.
    """
    def __init__(self, redis: Redis, ttl: int = DEFAULT_TTL):
        """
        Initialize the ResponseCache.
        
        redis: Redis connection
        ttl: How long to keep answers (default: 15 minutes)
        """
        self.redis = redis
        self.ttl = ttl
    
    @staticmethod
    def _make_hash(payload: dict) -> str:
        """
        Creates a unique "fingerprint" of the user's QUESTION (not the answer!).
        
        CORRECT FLOW:
        1. Question: {"message": "What's 2+2?"}
        2. Convert to hash: "abc123..." (unique fingerprint)
        3. Use hash as part of key to find/store answer
        
        NOTE: We hash the QUESTION, NOT the answer!
        - Question → Hashed → Used in key
        - Answer → Stored as JSON → Retrieved when key matches
        
        Example:
          Question: {"message": "Hello"} → hash: "abc123..."
          Question: {"message": "Hello"} → hash: "abc123..." (same question = same hash)
          Question: {"message": "Hi"} → hash: "xyz789..." (different question = different hash)
        """
        # Convert question to JSON string (sorted so same question = same string)
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        # Create unique fingerprint (hash)
        return hashlib.sha256(raw).hexdigest()

    def _build_key(
        self,
        user_id: str,
        model: str,
        payload: dict,
    ) -> str:
        """
        Creates a unique "address" to store/find the cached answer.
        
        SIMPLE EXPLANATION:
        - Combines: user_id + model_name + question_hash
        - Creates a unique key like: "response:gpt4:user123:abc456"
        - This key is like an "address" in Redis where we store the answer
        
        Example:
          user_id="user123", model="gpt4", question="Hello"
          → key: "response:gpt4:user123:abc123..."
        """
        payload_hash = self._make_hash(payload)
        return f"response:{model}:{user_id}:{payload_hash}"

    async def get(
        self,
        user_id: str,
        model: str,
        payload: dict,
    ) -> Optional[dict]:
        """
        Check if we already have an answer for this question.
        
        CORRECT FLOW:
        1. Hash the QUESTION (payload) → "abc123..."
        2. Build key: "response:gpt4:user123:abc123..."
        3. Check Redis: "Do I have an answer stored at this key?"
        4. If YES → Return the saved answer (stored as JSON, NOT hashed)
        5. If NO → Return None (need to ask LLM)
        
        Example:
          User asks: "What's the weather?"
          → Hash question: "abc123..."
          → Build key: "response:gpt4:user123:abc123..."
          → Check cache: Found answer "It's sunny" (stored as JSON)
          → Return: {"reply": "It's sunny"} (NOT hashed!)
        """
        key = self._build_key(user_id, model, payload)
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)  # Found it! Return saved answer
        return None  # Not found, need to ask LLM

    async def set(
        self,
        user_id: str,
        model: str,
        payload: dict,
        response: dict,
    ) -> None:
        """
        Save the LLM's answer so we can reuse it later.
        
        CORRECT FLOW:
        1. Hash the QUESTION (payload) → "abc123..."
        2. Build key: "response:gpt4:user123:abc123..."
        3. Store answer as JSON (NOT hashed!) at this key
        4. Set expiration time (15 minutes)
        5. Next time same question is asked → instant answer!
        
        IMPORTANT: We store the answer as JSON, NOT as a hash!
        - Question → Hashed → Used in key
        - Answer → Stored as JSON → Can be retrieved and read
        
        Example:
          Question: "What's the weather?" → hash: "abc123..."
          LLM answered: {"reply": "The weather is sunny"}
          → Key: "response:gpt4:user123:abc123..."
          → Store: Key = JSON string of answer (readable!)
          → Next time same question → instant answer!
        """
        key = self._build_key(user_id, model, payload)
        await self.redis.setex(
            key,
            self.ttl,  # Expire after 15 minutes
            json.dumps(response)  # Save the answer
        )

    async def invalidate_user(self, user_id: str) -> None:
        """
        Delete all cached answers for a specific user.
        
        SIMPLE EXPLANATION:
        - Finds all answers stored for this user
        - Deletes them all
        - Like erasing a user's page in the notebook
        
        Example:
          User wants to clear their cache
          → Find all keys like "response:*:user123:*"
          → Delete them all
          → User's cache is cleared
        """
        pattern = f"response:*:{user_id}:*"  # Find all keys for this user
        async for key in self.redis.scan_iter(match=pattern):
            await self.redis.delete(key)  # Delete each one
