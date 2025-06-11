"""
Conversation Memory for AI-to-AI Multi-turn Discussions

This module provides conversation persistence and context reconstruction for
stateless MCP environments. It enables multi-turn conversations between Claude
and Gemini by storing conversation state in SQLite across independent request cycles.

Key Features:
- UUID-based conversation thread identification
- Turn-by-turn conversation history storage
- Automatic turn limiting to prevent runaway conversations
- Context reconstruction for stateless request continuity
- SQLite-based persistence with automatic expiration cleanup
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

# Configuration constants
MAX_CONVERSATION_TURNS = 10  # Maximum turns allowed per conversation thread


class ConversationTurn(BaseModel):
    """Single turn in a conversation"""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    follow_up_question: Optional[str] = None
    files: Optional[list[str]] = None  # Files referenced in this turn
    tool_name: Optional[str] = None  # Tool used for this turn


class ThreadContext(BaseModel):
    """Complete conversation context"""

    thread_id: str
    created_at: str
    last_updated_at: str
    tool_name: str
    turns: list[ConversationTurn]
    initial_context: dict[str, Any]


def get_db_path() -> Path:
    """Get SQLite database path from environment or default location"""
    db_path = os.getenv("GEMINI_MCP_DB_PATH")
    if db_path:
        return Path(db_path)
    
    # Default to user's home directory or temp directory
    home_dir = Path.home()
    if home_dir.exists() and os.access(home_dir, os.W_OK):
        return home_dir / ".gemini_mcp_conversations.db"
    else:
        # Fallback to temp directory
        import tempfile
        return Path(tempfile.gettempdir()) / "gemini_mcp_conversations.db"


def init_database():
    """Initialize SQLite database with conversation tables"""
    db_path = get_db_path()
    
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_threads (
                thread_id TEXT PRIMARY KEY,
                context_json TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        
        # Index for cleanup operations
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at 
            ON conversation_threads(expires_at)
        """)
        
        conn.commit()


def cleanup_expired_threads():
    """Remove expired conversation threads"""
    db_path = get_db_path()
    if not db_path.exists():
        return
        
    with sqlite3.connect(db_path) as conn:
        now = datetime.now(timezone.utc)
        conn.execute(
            "DELETE FROM conversation_threads WHERE expires_at < ?",
            (now.isoformat(),)
        )
        conn.commit()


def get_connection():
    """Get SQLite connection with automatic initialization and cleanup"""
    init_database()
    cleanup_expired_threads()  # Clean up on each connection
    return sqlite3.connect(get_db_path())


def create_thread(tool_name: str, initial_request: dict[str, Any]) -> str:
    """Create new conversation thread and return thread ID"""
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=1)  # 1 hour TTL

    # Filter out non-serializable parameters
    filtered_context = {
        k: v
        for k, v in initial_request.items()
        if k not in ["temperature", "thinking_mode", "model", "continuation_id"]
    }

    context = ThreadContext(
        thread_id=thread_id,
        created_at=now.isoformat(),
        last_updated_at=now.isoformat(),
        tool_name=tool_name,
        turns=[],
        initial_context=filtered_context,
    )

    # Store in SQLite with 1 hour TTL
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conversation_threads 
            (thread_id, context_json, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (thread_id, context.model_dump_json(), now.isoformat(), expires_at.isoformat()))
        conn.commit()

    return thread_id


def get_thread(thread_id: str) -> Optional[ThreadContext]:
    """Retrieve thread context from SQLite"""
    if not thread_id or not _is_valid_uuid(thread_id):
        return None

    try:
        with get_connection() as conn:
            cursor = conn.execute("""
                SELECT context_json FROM conversation_threads 
                WHERE thread_id = ? AND expires_at > ?
            """, (thread_id, datetime.now(timezone.utc).isoformat()))
            
            row = cursor.fetchone()
            if row:
                return ThreadContext.model_validate_json(row[0])
            return None
    except Exception:
        return None


def add_turn(
    thread_id: str,
    role: str,
    content: str,
    follow_up_question: Optional[str] = None,
    files: Optional[list[str]] = None,
    tool_name: Optional[str] = None,
) -> bool:
    """Add turn to existing thread"""
    context = get_thread(thread_id)
    if not context:
        return False

    # Check turn limit
    if len(context.turns) >= MAX_CONVERSATION_TURNS:
        return False

    # Add new turn
    turn = ConversationTurn(
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        follow_up_question=follow_up_question,
        files=files,
        tool_name=tool_name,
    )

    context.turns.append(turn)
    context.last_updated_at = datetime.now(timezone.utc).isoformat()

    # Save back to SQLite with refreshed TTL
    try:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=1)  # Refresh TTL
        
        with get_connection() as conn:
            conn.execute("""
                UPDATE conversation_threads 
                SET context_json = ?, expires_at = ?
                WHERE thread_id = ?
            """, (context.model_dump_json(), expires_at.isoformat(), thread_id))
            conn.commit()
        return True
    except Exception:
        return False


def build_conversation_history(context: ThreadContext) -> str:
    """Build formatted conversation history"""
    if not context.turns:
        return ""

    history_parts = [
        "=== CONVERSATION HISTORY ===",
        f"Thread: {context.thread_id}",
        f"Tool: {context.tool_name}",
        f"Turn {len(context.turns)}/{MAX_CONVERSATION_TURNS}",
        "",
        "Previous exchanges:",
    ]

    for i, turn in enumerate(context.turns, 1):
        role_label = "Claude" if turn.role == "user" else "Gemini"

        # Add turn header with tool info if available
        turn_header = f"\n--- Turn {i} ({role_label}"
        if turn.tool_name:
            turn_header += f" using {turn.tool_name}"
        turn_header += ") ---"
        history_parts.append(turn_header)

        # Add files context if present
        if turn.files:
            history_parts.append(f"📁 Files referenced: {', '.join(turn.files)}")
            history_parts.append("")  # Empty line for readability

        # Add the actual content
        history_parts.append(turn.content)

        # Add follow-up question if present
        if turn.follow_up_question:
            history_parts.append(f"\n[Gemini's Follow-up: {turn.follow_up_question}]")

    history_parts.extend(
        ["", "=== END HISTORY ===", "", "Continue this conversation by building on the previous context."]
    )

    return "\n".join(history_parts)


def _is_valid_uuid(val: str) -> bool:
    """Validate UUID format for security"""
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False
