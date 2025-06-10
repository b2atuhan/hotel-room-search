from typing import Any, Optional
from functools import lru_cache

class CacheManager:
    def __init__(self):
        """Initialize the cache manager with a simple memory cache."""
        self.memory_cache = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get a value from memory cache."""
        return self.memory_cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Store a value in memory cache."""
        self.memory_cache[key] = value
    
    def clear(self) -> None:
        """Clear the memory cache."""
        self.memory_cache.clear()
    
    def remove(self, key: str) -> None:
        """Remove a specific key from memory cache."""
        self.memory_cache.pop(key, None) 