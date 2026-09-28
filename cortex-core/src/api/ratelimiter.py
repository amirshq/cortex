# this is the token bucket rate limiter implementation
import time

"""
TokenBucket Rate Limiter - Explanation of Variables:

Think of a TokenBucket like a water bucket with a leaky faucet:

┌─────────────────┐
│  🪙🪙🪙🪙🪙      │  ← Bucket (capacity)
│  🪙🪙🪙🪙🪙      │     Current tokens inside
│  🪙🪙🪙🪙🪙      │
└─────────────────┘
       ↓ (refill_rate)
   🚰 Drip... Drip...

VARIABLES EXPLAINED:

1. capacity (int)
   - Maximum number of tokens the bucket can hold
   - Example: capacity = 10 means bucket can hold max 10 tokens
   - Like the size of the bucket - once full, no more tokens can be added
   - Prevents unlimited token accumulation

2. refill_rate (float)
   - Number of tokens added per second
   - Example: refill_rate = 2.0 means 2 tokens per second
   - Like the speed of the faucet dripping into the bucket
   - Continuous refill over time (not all at once)

3. tokens (int)
   - Current number of tokens available in the bucket
   - Starts at capacity (full bucket)
   - Decreases when consumed
   - Increases when refilled (up to capacity)
   - Like the current water level in the bucket

4. last_refill_timestamp (float)
   - Unix timestamp (seconds since epoch) of the last refill
   - Used to calculate how much time has passed
   - Example: 1704067200.0 (January 1, 2024, 00:00:00 UTC)
   - Updated every time _refill() is called
   - Like a stopwatch to track time between refills

5. elapsed (float) - calculated in _refill()
   - Time difference in seconds between now and last_refill_timestamp
   - Formula: elapsed = now - last_refill_timestamp
   - Example: If 2.5 seconds passed, elapsed = 2.5
   - Used to calculate how many tokens to add

6. added_tokens (float) - calculated in _refill()
   - Number of tokens to add based on elapsed time
   - Formula: added_tokens = elapsed * refill_rate
   - Example: elapsed=2.5, refill_rate=2.0 → added_tokens = 5.0
   - Can't exceed capacity (capped by min() function)

EXAMPLE SCENARIO:

Initialization:
  capacity = 10
  refill_rate = 2.0 tokens/second
  tokens = 10 (bucket starts full)
  last_refill_timestamp = 1000.0

After 3 seconds pass, user tries to consume 5 tokens:
  
  Step 1: _refill() is called
    now = 1003.0
    elapsed = 1003.0 - 1000.0 = 3.0 seconds
    added_tokens = 3.0 * 2.0 = 6.0 tokens
    tokens = min(10, 10 + 6.0) = 10 (capped at capacity)
    last_refill_timestamp = 1003.0
  
  Step 2: consume(5) is called
    tokens (10) >= 5? Yes!
    tokens = 10 - 5 = 5
    Return True (request allowed)

After 2 more seconds, user tries to consume 8 tokens:
  
  Step 1: _refill() is called
    now = 1005.0
    elapsed = 1005.0 - 1003.0 = 2.0 seconds
    added_tokens = 2.0 * 2.0 = 4.0 tokens
    tokens = min(10, 5 + 4.0) = 9
    last_refill_timestamp = 1005.0
  
  Step 2: consume(8) is called
    tokens (9) >= 8? Yes!
    tokens = 9 - 8 = 1
    Return True (request allowed)

User immediately tries to consume 5 tokens:
  
  Step 1: _refill() is called
    now = 1005.1
    elapsed = 1005.1 - 1005.0 = 0.1 seconds
    added_tokens = 0.1 * 2.0 = 0.2 tokens
    tokens = min(10, 1 + 0.2) = 1.2
  
  Step 2: consume(5) is called
    tokens (1.2) >= 5? No!
    Return False (request denied - rate limited!)
"""

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize the TokenBucket.

        :param capacity: Maximum number of tokens in the bucket.
                        Example: 10 means bucket can hold max 10 tokens
        :param refill_rate: Number of tokens to add per second.
                          Example: 2.0 means 2 tokens per second
        """
        self.capacity = capacity  # Max tokens bucket can hold
        self.refill_rate = refill_rate  # Tokens added per second
        self.tokens = capacity  # Current tokens (starts full)
        self.last_refill_timestamp = time.time()  # Last refill time

    def _refill(self):
        """
        Refill tokens in the bucket based on the elapsed time since the last refill.
        
        How it works:
        1. Calculate elapsed time (now - last_refill_timestamp)
        2. Calculate tokens to add (elapsed * refill_rate)
        3. Add tokens (capped at capacity)
        4. Update last_refill_timestamp
        """
        now = time.time()  # Current time in seconds
        elapsed = now - self.last_refill_timestamp  # Time passed since last refill
        added_tokens = elapsed * self.refill_rate  # Tokens to add based on time
        self.tokens = min(self.capacity, self.tokens + added_tokens)  # Add tokens (max = capacity)
        self.last_refill_timestamp = now  # Update timestamp for next calculation

    def consume(self, tokens: int) -> bool:
        """
        Attempt to consume a specified number of tokens from the bucket.

        :param tokens: Number of tokens to consume (usually 1 per request).
        :return: True if tokens were successfully consumed, False otherwise (rate limited).
        
        Process:
        1. Refill tokens first (based on time passed)
        2. Check if enough tokens available
        3. If yes, consume tokens and return True
        4. If no, return False (rate limit exceeded)
        """
        self._refill()  # Refill before checking
        if self.tokens >= tokens:
            self.tokens -= tokens  # Consume tokens
            return True  # Request allowed
        return False  # Request denied (rate limited)