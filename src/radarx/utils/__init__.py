"""Utility functions and helpers."""

from datetime import datetime
from typing import Any, Dict
import hashlib
import json


def generate_id(prefix: str, *args) -> str:
    """Generate a unique ID from components."""
    components = "_".join(str(arg) for arg in args)
    hash_suffix = hashlib.md5(components.encode()).hexdigest()[:8]
    return f"{prefix}_{hash_suffix}"


def normalize_timestamp(ts: Any) -> datetime:
    """Normalize various timestamp formats to datetime."""
    if isinstance(ts, datetime):
        return ts
    elif isinstance(ts, str):
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    elif isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    else:
        raise ValueError(f"Cannot normalize timestamp: {ts}")


def serialize_for_cache(obj: Any) -> str:
    """Serialize object for caching."""
    if hasattr(obj, 'model_dump'):
        return json.dumps(obj.model_dump())
    return json.dumps(obj)


def deserialize_from_cache(data: str, model_class: type) -> Any:
    """Deserialize object from cache."""
    obj_dict = json.loads(data)
    if hasattr(model_class, 'model_validate'):
        return model_class.model_validate(obj_dict)
    return obj_dict
