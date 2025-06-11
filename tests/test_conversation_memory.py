"""
Test suite for conversation memory system

Tests the SQLite-based conversation persistence needed for AI-to-AI multi-turn
discussions in stateless MCP environments.
"""

import tempfile
import os
from unittest.mock import patch
from pathlib import Path

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


class TestConversationMemory:
    """Test the conversation memory system for stateless MCP requests"""

    def setup_method(self):
        """Set up temporary database for each test"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)

    def teardown_method(self):
        """Clean up temporary database after each test"""
        if self.db_path.exists():
            os.unlink(self.db_path)

    @patch("utils.conversation_memory.get_db_path")
    def test_create_thread(self, mock_db_path):
        """Test creating a new thread"""
        mock_db_path.return_value = self.db_path

        thread_id = create_thread("chat", {"prompt": "Hello", "files": ["/test.py"]})

        assert thread_id is not None
        assert len(thread_id) == 36  # UUID4 length

        # Verify thread was stored
        context = get_thread(thread_id)
        assert context is not None
        assert context.thread_id == thread_id
        assert context.tool_name == "chat"

    @patch("utils.conversation_memory.get_db_path")
    def test_get_thread_valid(self, mock_db_path):
        """Test retrieving an existing thread"""
        mock_db_path.return_value = self.db_path

        # Create a thread first
        thread_id = create_thread("chat", {"prompt": "test"})
        
        # Retrieve it
        context = get_thread(thread_id)

        assert context is not None
        assert context.thread_id == thread_id
        assert context.tool_name == "chat"

    @patch("utils.conversation_memory.get_db_path")
    def test_get_thread_invalid_uuid(self, mock_db_path):
        """Test retrieving thread with invalid UUID"""
        mock_db_path.return_value = self.db_path

        context = get_thread("invalid-uuid")
        assert context is None

    @patch("utils.conversation_memory.get_db_path")
    def test_get_thread_not_found(self, mock_db_path):
        """Test retrieving non-existent thread"""
        mock_db_path.return_value = self.db_path

        test_uuid = "12345678-1234-1234-1234-123456789012"
        context = get_thread(test_uuid)
        assert context is None

    @patch("utils.conversation_memory.get_db_path")
    def test_add_turn_success(self, mock_db_path):
        """Test adding a turn to existing thread"""
        mock_db_path.return_value = self.db_path

        # Create thread
        thread_id = create_thread("chat", {"prompt": "Hello"})
        
        # Add turn
        success = add_turn(
            thread_id, 
            "assistant", 
            "Hello there!", 
            follow_up_question="How can I help?"
        )
        
        assert success is True
        
        # Verify turn was added
        context = get_thread(thread_id)
        assert len(context.turns) == 1
        assert context.turns[0].role == "assistant"
        assert context.turns[0].content == "Hello there!"
        assert context.turns[0].follow_up_question == "How can I help?"

    @patch("utils.conversation_memory.get_db_path")
    def test_add_turn_max_limit(self, mock_db_path):
        """Test that turn limit is enforced"""
        mock_db_path.return_value = self.db_path

        # Create thread
        thread_id = create_thread("chat", {"prompt": "Hello"})
        
        # Add maximum turns
        for i in range(MAX_CONVERSATION_TURNS):
            success = add_turn(thread_id, "assistant", f"Response {i}")
            assert success is True
            
        # Try to add one more (should fail)
        success = add_turn(thread_id, "assistant", "Too many turns")
        assert success is False

    @patch("utils.conversation_memory.get_db_path")
    def test_add_turn_nonexistent_thread(self, mock_db_path):
        """Test adding turn to non-existent thread"""
        mock_db_path.return_value = self.db_path

        test_uuid = "12345678-1234-1234-1234-123456789012"
        success = add_turn(test_uuid, "assistant", "Response")
        assert success is False

    @patch("utils.conversation_memory.get_db_path")
    def test_build_conversation_history(self, mock_db_path):
        """Test building conversation history string"""
        mock_db_path.return_value = self.db_path

        # Create thread and add turns
        thread_id = create_thread("chat", {"prompt": "Hello"})
        add_turn(thread_id, "user", "Hello", files=["/test.py"])
        add_turn(thread_id, "assistant", "Hi there!", follow_up_question="Need help?")
        
        context = get_thread(thread_id)
        history = build_conversation_history(context)
        
        assert "CONVERSATION HISTORY" in history
        assert "Turn 2/5" in history
        assert "Claude" in history  # user role
        assert "Gemini" in history  # assistant role
        assert "Hello" in history
        assert "Hi there!" in history
        assert "Need help?" in history
        assert "/test.py" in history

    def test_build_conversation_history_empty(self):
        """Test building history for thread with no turns"""
        context = ThreadContext(
            thread_id="test",
            created_at="2023-01-01T00:00:00Z",
            last_updated_at="2023-01-01T00:00:00Z",
            tool_name="chat",
            turns=[],
            initial_context={}
        )
        
        history = build_conversation_history(context)
        assert history == ""


class TestConversationFlow:
    """Test complete conversation flows simulating stateless MCP requests"""

    def setup_method(self):
        """Set up temporary database for each test"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)

    def teardown_method(self):
        """Clean up temporary database after each test"""
        if self.db_path.exists():
            os.unlink(self.db_path)

    @patch("utils.conversation_memory.get_db_path")
    def test_complete_conversation_cycle(self, mock_db_path):
        """Test a complete 5-turn conversation until limit reached"""
        mock_db_path.return_value = self.db_path

        # REQUEST 1: Initial request creates thread
        thread_id = create_thread("chat", {"prompt": "Analyze this code"})
        
        # Add assistant response with follow-up
        success = add_turn(
            thread_id,
            "assistant",
            "Code analysis complete",
            follow_up_question="Would you like me to check error handling?",
        )
        assert success is True

        # REQUEST 2: User responds to follow-up (independent request cycle)
        success = add_turn(thread_id, "user", "Yes, check error handling")
        assert success is True

        success = add_turn(
            thread_id, "assistant", "Error handling reviewed", follow_up_question="Should I examine the test coverage?"
        )
        assert success is True

        # REQUEST 3-4: Continue conversation
        success = add_turn(thread_id, "user", "Yes, check tests")
        assert success is True

        success = add_turn(thread_id, "assistant", "Test coverage analyzed")
        assert success is True

        # Verify we have MAX_CONVERSATION_TURNS
        context = get_thread(thread_id)
        assert len(context.turns) == MAX_CONVERSATION_TURNS

        # REQUEST 5: Try to exceed MAX_CONVERSATION_TURNS limit - should fail
        success = add_turn(thread_id, "user", "This should be rejected")
        assert success is False  # CONVERSATION STOPS HERE

    @patch("utils.conversation_memory.get_db_path")
    def test_invalid_continuation_id_error(self, mock_db_path):
        """Test that invalid continuation IDs raise proper error for restart"""
        mock_db_path.return_value = self.db_path
        
        from server import reconstruct_thread_context

        arguments = {"continuation_id": "invalid-uuid-12345", "prompt": "Continue conversation"}

        # Should raise ValueError asking to restart
        with pytest.raises(ValueError) as exc_info:
            import asyncio
            asyncio.run(reconstruct_thread_context(arguments))

        error_msg = str(exc_info.value)
        assert "Conversation thread 'invalid-uuid-12345' was not found or has expired" in error_msg
        assert (
            "Please restart the conversation by providing your full question/prompt without the continuation_id"
            in error_msg
        )

    def test_dynamic_max_turns_configuration(self):
        """Test that all functions respect MAX_CONVERSATION_TURNS configuration"""
        # Test with different max values by checking current behavior
        test_values = [3, 7, 10]

        for test_max in test_values:
            # Create turns up to the test limit
            turns = [
                ConversationTurn(role="user", content=f"Turn {i}", timestamp="2023-01-01T00:00:00Z")
                for i in range(test_max)
            ]

            # Test history building respects the limit
            test_uuid = "12345678-1234-1234-1234-123456789012"
            context = ThreadContext(
                thread_id=test_uuid,
                created_at="2023-01-01T00:00:00Z",
                last_updated_at="2023-01-01T00:00:00Z",
                tool_name="chat",
                turns=turns,
                initial_context={},
            )

            history = build_conversation_history(context)
            expected_turn_text = f"Turn {test_max}/{MAX_CONVERSATION_TURNS}"
            assert expected_turn_text in history

    @patch("utils.conversation_memory.get_db_path")
    def test_complete_conversation_with_dynamic_turns(self, mock_db_path):
        """Test complete conversation respecting MAX_CONVERSATION_TURNS dynamically"""
        mock_db_path.return_value = self.db_path

        thread_id = create_thread("chat", {"prompt": "Start conversation"})

        # Simulate conversation up to MAX_CONVERSATION_TURNS - 1
        for turn_num in range(MAX_CONVERSATION_TURNS - 1):
            # Should succeed
            success = add_turn(thread_id, "user", f"User turn {turn_num + 1}")
            assert success is True, f"Turn {turn_num + 1} should succeed"

        # Now we should be at the limit
        context = get_thread(thread_id)
        assert len(context.turns) == MAX_CONVERSATION_TURNS - 1

        # Add one more to hit the limit
        success = add_turn(thread_id, "assistant", "Final turn")
        assert success is True

        # This should fail - at the limit
        success = add_turn(thread_id, "user", "This should fail")
        assert success is False, f"Turn {MAX_CONVERSATION_TURNS + 1} should fail"

    @patch("utils.conversation_memory.get_db_path")
    def test_conversation_with_files_and_context_preservation(self, mock_db_path):
        """Test complete conversation flow with file tracking and context preservation"""
        mock_db_path.return_value = self.db_path

        # Start conversation with files
        thread_id = create_thread("analyze", {"prompt": "Analyze this codebase", "files": ["/project/src/"]})

        # Add Gemini's response with follow-up
        success = add_turn(
            thread_id,
            "assistant",
            "I've analyzed your codebase structure.",
            follow_up_question="Would you like me to examine the test coverage?",
            files=["/project/src/main.py", "/project/src/utils.py"],
            tool_name="analyze",
        )
        assert success is True

        # User responds with test files
        success = add_turn(
            thread_id, "user", "Yes, check the test coverage", files=["/project/tests/", "/project/test_main.py"]
        )
        assert success is True

        # Gemini analyzes tests
        success = add_turn(
            thread_id,
            "assistant",
            "Test coverage analysis complete. Coverage is 85%.",
            files=["/project/tests/test_utils.py", "/project/coverage.html"],
            tool_name="analyze",
        )
        assert success is True

        # Build conversation history and verify chronological file preservation
        context = get_thread(thread_id)
        history = build_conversation_history(context)

        # Verify chronological order and speaker identification
        assert "--- Turn 1 (Gemini using analyze) ---" in history
        assert "--- Turn 2 (Claude) ---" in history
        assert "--- Turn 3 (Gemini using analyze) ---" in history

        # Verify all files are preserved in chronological order
        turn_1_files = "📁 Files referenced: /project/src/main.py, /project/src/utils.py"
        turn_2_files = "📁 Files referenced: /project/tests/, /project/test_main.py"
        turn_3_files = "📁 Files referenced: /project/tests/test_utils.py, /project/coverage.html"

        assert turn_1_files in history
        assert turn_2_files in history
        assert turn_3_files in history

        # Verify content and follow-ups
        assert "I've analyzed your codebase structure." in history
        assert "Yes, check the test coverage" in history
        assert "Test coverage analysis complete. Coverage is 85%." in history
        assert "[Gemini's Follow-up: Would you like me to examine the test coverage?]" in history

    @patch("utils.conversation_memory.get_db_path")
    def test_follow_up_question_parsing_cycle(self, mock_db_path):
        """Test follow-up question persistence across request cycles"""
        mock_db_path.return_value = self.db_path

        thread_id = create_thread("debug", {"prompt": "Debug this error"})

        # First cycle: Assistant generates follow-up
        success = add_turn(
            thread_id,
            "assistant",
            "Found potential issue in authentication",
            follow_up_question="Should I examine the authentication middleware?",
        )
        assert success is True

        # Second cycle: Retrieve conversation history
        context = get_thread(thread_id)

        # Build history to verify follow-up is preserved
        history = build_conversation_history(context)
        assert "Found potential issue in authentication" in history
        assert "[Gemini's Follow-up: Should I examine the authentication middleware?]" in history

    @patch("utils.conversation_memory.get_db_path")
    def test_stateless_request_isolation(self, mock_db_path):
        """Test that each request cycle is independent but shares context via SQLite"""
        mock_db_path.return_value = self.db_path

        # Simulate two different "processes" accessing same thread
        thread_id = create_thread("thinkdeep", {"prompt": "Think about architecture"})

        # Process 1: Creates thread and adds turn
        success = add_turn(
            thread_id, "assistant", "Architecture analysis", follow_up_question="Want to explore scalability?"
        )
        assert success is True

        # Process 2: Different "request cycle" accesses same thread
        # Verify context continuity across "processes"
        retrieved_context = get_thread(thread_id)
        assert retrieved_context is not None
        assert len(retrieved_context.turns) == 1
        assert retrieved_context.turns[0].follow_up_question == "Want to explore scalability?"


class TestFollowUpInstructions:
    """Test follow-up instruction generation"""

    def test_get_follow_up_instructions_basic(self):
        """Test basic follow-up instruction generation"""
        # current_turn_count=1, max_turns=5 -> remaining = 5-1-1 = 3
        current_turn_count = 1
        instructions = get_follow_up_instructions(current_turn_count)
        
        assert "CONVERSATION THREADING" in instructions
        assert "3 exchanges remaining" in instructions
        assert "follow_up_question" in instructions
        assert "suggested_params" in instructions

    def test_get_follow_up_instructions_final_turn(self):
        """Test follow-up instructions for final turn"""
        # current_turn_count=4, max_turns=5 -> remaining = 5-4-1 = 0, should hit final turn logic
        current_turn_count = 4
        instructions = get_follow_up_instructions(current_turn_count)
        
        assert "final exchange" in instructions or "IMPORTANT" in instructions
        # Final turn doesn't show "CONVERSATION THREADING"

    def test_follow_up_instructions_dynamic_behavior(self):
        """Test that follow-up instructions change correctly based on turn count and max setting"""
        # Test with default MAX_CONVERSATION_TURNS
        max_turns = MAX_CONVERSATION_TURNS

        # Test early conversation (should allow follow-ups)
        early_instructions = get_follow_up_instructions(0, max_turns)
        assert "CONVERSATION THREADING" in early_instructions
        assert f"({max_turns - 1} exchanges remaining)" in early_instructions

        # Test mid conversation
        mid_instructions = get_follow_up_instructions(2, max_turns)
        assert "CONVERSATION THREADING" in mid_instructions
        assert f"({max_turns - 3} exchanges remaining)" in mid_instructions

        # Test approaching limit (should stop follow-ups)
        limit_instructions = get_follow_up_instructions(max_turns - 1, max_turns)
        assert "Do NOT include any follow-up questions" in limit_instructions
        assert "CONVERSATION THREADING" not in limit_instructions

        # Test with custom max_turns to ensure dynamic behavior
        custom_max = 3
        custom_early = get_follow_up_instructions(0, custom_max)
        assert f"({custom_max - 1} exchanges remaining)" in custom_early

        custom_limit = get_follow_up_instructions(custom_max - 1, custom_max)
        assert "Do NOT include any follow-up questions" in custom_limit

    def test_follow_up_instructions_defaults_to_config(self):
        """Test that follow-up instructions use MAX_CONVERSATION_TURNS when max_turns not provided"""
        instructions = get_follow_up_instructions(0)  # No max_turns parameter
        expected_remaining = MAX_CONVERSATION_TURNS - 1
        assert f"({expected_remaining} exchanges remaining)" in instructions