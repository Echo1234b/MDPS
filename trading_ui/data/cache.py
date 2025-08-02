import redis
import json
from typing import Any, Optional

class Cache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)

    def set(self, key: str, value: Any, expiration: Optional[int] = None):
        """Set value in cache with optional expiration"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        self.redis.set(key, value, ex=expiration)

    def get(self, key: str) -> Any:
        """Get value from cache"""
        value = self.redis.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.decode('utf-8')

    def delete(self, key: str):
        """Delete value from cache"""
        self.redis.delete(key)

    def clear(self):
        """Clear all values from cache"""
        self.redis.flushdb()
