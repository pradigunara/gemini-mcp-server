"""
Test suite for conversation memory system WITH continuation offers enabled

Tests the complete conversation threading system including SQLite persistence
and continuation offer generation in realistic usage scenarios.
"""

import tempfile
import os
from unittest.mock import patch
from pathlib import Path
import json

import pytest

from server import get_follow_up_instructions
from utils.conversation_memory import (
    MAX_CONVERSATION_TURNS,
    ConversationTurn,
    ThreadContext,
    add_turn,
    build_conversation_history,
    create_thread,
    get_thread,
)
from tools.chat import ChatTool


class TestConversationWithContinuation:
    """Test conversation memory with continuation offers enabled (realistic usage)"""

    def setup_method(self):
        """Set up temporary database for each test"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)

    def teardown_method(self):
        """Clean up temporary database after each test"""
        if self.db_path.exists():
            os.unlink(self.db_path)

    @pytest.mark.asyncio
    @patch("utils.conversation_memory.get_db_path")
    async def test_new_conversation_creates_continuation_offer(self, mock_db_path):
        """Test that new conversations create continuation offers with SQLite threads"""
        mock_db_path.return_value = self.db_path
        
        # Create a chat tool request (new conversation)
        chat_tool = ChatTool()
        
        request_data = {
            "prompt": "Explain how async/await works in Python"
        }
        
        # Execute the tool (this should create a continuation offer)
        with patch.object(chat_tool, 'create_model') as mock_create_model:
            # Create proper mock response structure
            mock_part = type('MockPart', (), {'text': "Python async/await allows you to write asynchronous code..."})() 
            mock_content = type('MockContent', (), {'parts': [mock_part]})()
            mock_candidate = type('MockCandidate', (), {'content': mock_content})()
            mock_response = type('MockResponse', (), {'candidates': [mock_candidate]})()
            
            mock_model = type('MockModel', (), {
                'generate_content': lambda self, prompt, **kwargs: mock_response
            })()
            mock_create_model.return_value = mock_model
            
            result = await chat_tool.execute(request_data)
        
        # Parse the response
        response_data = json.loads(result[0].text)
        
        # Should have continuation_available status
        assert response_data["status"] == "continuation_available"
        assert "continuation_offer" in response_data
        
        # Check continuation offer details
        continuation_offer = response_data["continuation_offer"]
        assert "continuation_id" in continuation_offer
        assert "remaining_turns" in continuation_offer
        assert continuation_offer["remaining_turns"] == MAX_CONVERSATION_TURNS - 1
        
        # Verify SQLite thread was created
        thread_id = continuation_offer["continuation_id"]
        context = get_thread(thread_id)
        assert context is not None
        assert context.tool_name == "chat"
        assert len(context.turns) == 1  # Assistant's response was added
        assert context.turns[0].role == "assistant"

    @pytest.mark.asyncio
    @patch("utils.conversation_memory.get_db_path")
    async def test_continued_conversation_uses_history(self, mock_db_path):
        """Test that continuation requests use conversation history from SQLite"""
        mock_db_path.return_value = self.db_path
        
        # Step 1: Create initial conversation thread manually
        thread_id = create_thread("chat", {"prompt": "Explain async/await"})
        add_turn(thread_id, "assistant", "Python async/await allows asynchronous programming...")
        add_turn(thread_id, "user", "Can you give me an example?")
        
        # Step 2: Create continuation request  
        chat_tool = ChatTool()
        
        request_data = {
            "prompt": "Show me a practical example",
            "continuation_id": thread_id
        }
        
        # Execute with continuation ID
        with patch.object(chat_tool, 'create_model') as mock_create_model:
            # Capture the prompt sent to Gemini to verify history is included
            captured_prompt = None
            
            def capture_prompt(self, prompt, **kwargs):
                nonlocal captured_prompt
                captured_prompt = prompt
                # Create proper mock response structure
                mock_part = type('MockPart', (), {'text': "Here's a practical async example with aiohttp..."})() 
                mock_content = type('MockContent', (), {'parts': [mock_part]})()
                mock_candidate = type('MockCandidate', (), {'content': mock_content})()
                return type('MockResponse', (), {'candidates': [mock_candidate]})()
            
            mock_model = type('MockModel', (), {
                'generate_content': capture_prompt
            })()
            mock_create_model.return_value = mock_model
            
            result = await chat_tool.execute(request_data)
        
        # Verify conversation history was included in prompt
        assert "CONVERSATION HISTORY" in captured_prompt
        assert "Thread:" in captured_prompt
        assert "Python async/await allows asynchronous programming" in captured_prompt
        assert "Can you give me an example?" in captured_prompt
        
        # Verify response format for continuation
        response_data = json.loads(result[0].text)
        
        # Continued conversations can be either success or requires_continuation
        # depending on whether Gemini asks a follow-up
        assert response_data["status"] in ["success", "requires_continuation"]

    @patch("utils.conversation_memory.get_db_path")
    def test_conversation_turn_limit_with_continuation(self, mock_db_path):
        """Test that conversation turn limits work with continuation system"""
        mock_db_path.return_value = self.db_path
        
        # Create a thread and fill it to the limit
        thread_id = create_thread("chat", {"prompt": "Start conversation"})
        
        # Add turns up to the limit
        for i in range(MAX_CONVERSATION_TURNS):
            success = add_turn(thread_id, "user" if i % 2 == 0 else "assistant", f"Turn {i + 1}")
            assert success is True
        
        # Try to add one more turn (should fail)
        success = add_turn(thread_id, "user", "This should fail")
        assert success is False
        
        # Verify conversation is at limit
        context = get_thread(thread_id)
        assert len(context.turns) == MAX_CONVERSATION_TURNS

    @pytest.mark.asyncio
    @patch("utils.conversation_memory.get_db_path") 
    async def test_continuation_offer_metadata(self, mock_db_path):
        """Test that continuation offers include correct metadata"""
        mock_db_path.return_value = self.db_path
        
        chat_tool = ChatTool()
        
        request_data = {
            "prompt": "What is machine learning?",
            "files": ["/project/ml_model.py"]
        }
        
        with patch.object(chat_tool, 'create_model') as mock_create_model:
            # Create proper mock response structure
            mock_part = type('MockPart', (), {'text': "Machine learning is a subset of AI..."})() 
            mock_content = type('MockContent', (), {'parts': [mock_part]})()
            mock_candidate = type('MockCandidate', (), {'content': mock_content})()
            mock_response = type('MockResponse', (), {'candidates': [mock_candidate]})()
            
            mock_model = type('MockModel', (), {
                'generate_content': lambda self, prompt, **kwargs: mock_response
            })()
            mock_create_model.return_value = mock_model
            
            result = await chat_tool.execute(request_data)
        
        response_data = json.loads(result[0].text)
        
        # Check metadata
        assert "metadata" in response_data
        metadata = response_data["metadata"]
        assert metadata["tool_name"] == "chat"
        assert "thread_id" in metadata
        assert metadata["remaining_turns"] == MAX_CONVERSATION_TURNS - 1
        
        # Verify files were preserved in SQLite
        thread_id = metadata["thread_id"]
        context = get_thread(thread_id)
        assert context.initial_context["files"] == ["/project/ml_model.py"]

    @pytest.mark.asyncio
    async def test_continuation_system_can_be_disabled(self):
        """Test that continuation system respects DISABLE_CONTINUATION_OFFERS"""
        
        # Test with environment variable set
        with patch.dict(os.environ, {"DISABLE_CONTINUATION_OFFERS": "true"}):
            chat_tool = ChatTool()
            
            request_data = {
                "prompt": "Simple question"
            }
            
            with patch.object(chat_tool, 'create_model') as mock_create_model:
                # Create proper mock response structure
                mock_part = type('MockPart', (), {'text': "Simple answer"})()
                mock_content = type('MockContent', (), {'parts': [mock_part]})()
                mock_candidate = type('MockCandidate', (), {'content': mock_content})()
                mock_response = type('MockResponse', (), {'candidates': [mock_candidate]})()
                
                mock_model = type('MockModel', (), {
                    'generate_content': lambda self, prompt, **kwargs: mock_response
                })()
                mock_create_model.return_value = mock_model
                
                result = await chat_tool.execute(request_data)
            
            response_data = json.loads(result[0].text)
            
            # Should be simple success, no continuation offer
            assert response_data["status"] == "success"
            assert response_data["continuation_offer"] is None

    @patch("utils.conversation_memory.get_db_path")
    def test_sqlite_persistence_across_sessions(self, mock_db_path):
        """Test that SQLite provides persistence across different 'sessions'"""
        mock_db_path.return_value = self.db_path
        
        # Session 1: Create conversation
        thread_id = create_thread("analyze", {"prompt": "Analyze this code", "files": ["/app.py"]})
        add_turn(thread_id, "assistant", "Code analysis shows...", follow_up_question="Want details?")
        
        # Simulate new session/process (re-connect to same database)
        context = get_thread(thread_id)
        assert context is not None
        assert len(context.turns) == 1
        assert context.turns[0].follow_up_question == "Want details?"
        assert context.initial_context["files"] == ["/app.py"]
        
        # Session 2: Continue conversation
        add_turn(thread_id, "user", "Yes, show me details")
        add_turn(thread_id, "assistant", "Here are the detailed findings...")
        
        # Session 3: Verify persistence
        final_context = get_thread(thread_id)
        assert len(final_context.turns) == 3
        assert final_context.turns[2].content == "Here are the detailed findings..."

    @patch("utils.conversation_memory.get_db_path")
    def test_multiple_concurrent_conversations(self, mock_db_path):
        """Test SQLite handles multiple concurrent conversations"""
        mock_db_path.return_value = self.db_path
        
        # Create multiple conversation threads
        thread1 = create_thread("chat", {"prompt": "Question 1"})
        thread2 = create_thread("analyze", {"prompt": "Question 2"})
        thread3 = create_thread("debug", {"prompt": "Question 3"})
        
        # Add turns to each thread
        add_turn(thread1, "assistant", "Answer 1")
        add_turn(thread2, "assistant", "Answer 2") 
        add_turn(thread3, "assistant", "Answer 3")
        
        # Verify each thread maintains separate state
        context1 = get_thread(thread1)
        context2 = get_thread(thread2)
        context3 = get_thread(thread3)
        
        assert context1.tool_name == "chat"
        assert context2.tool_name == "analyze"
        assert context3.tool_name == "debug"
        
        assert context1.turns[0].content == "Answer 1"
        assert context2.turns[0].content == "Answer 2"
        assert context3.turns[0].content == "Answer 3"
        
        # Cross-contamination check
        assert len(context1.turns) == 1
        assert len(context2.turns) == 1 
        assert len(context3.turns) == 1