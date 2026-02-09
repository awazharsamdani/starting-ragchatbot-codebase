from typing import Dict, List, Optional
from models import create_message

# Module-level state for sessions
_sessions: Dict[str, List[dict]] = {}
_session_counter = 0
_max_history = 5

def initialize_session_manager(max_history: int = 5):
    """Initialize the session manager with configuration"""
    global _max_history
    _max_history = max_history

def create_session() -> str:
    """Create a new conversation session"""
    global _session_counter
    _session_counter += 1
    session_id = f"session_{_session_counter}"
    _sessions[session_id] = []
    return session_id

def add_message(session_id: str, role: str, content: str):
    """Add a message to the conversation history"""
    if session_id not in _sessions:
        _sessions[session_id] = []

    message = create_message(role=role, content=content)
    _sessions[session_id].append(message)

    # Keep conversation history within limits
    if len(_sessions[session_id]) > _max_history * 2:
        _sessions[session_id] = _sessions[session_id][-_max_history * 2:]

def add_exchange(session_id: str, user_message: str, assistant_message: str):
    """Add a complete question-answer exchange"""
    add_message(session_id, "user", user_message)
    add_message(session_id, "assistant", assistant_message)

def get_conversation_history(session_id: Optional[str]) -> Optional[str]:
    """Get formatted conversation history for a session"""
    if not session_id or session_id not in _sessions:
        return None

    messages = _sessions[session_id]
    if not messages:
        return None

    # Format messages for context
    formatted_messages = []
    for msg in messages:
        formatted_messages.append(f"{msg['role'].title()}: {msg['content']}")

    return "\n".join(formatted_messages)

def clear_session(session_id: str):
    """Clear all messages from a session"""
    if session_id in _sessions:
        _sessions[session_id] = []
