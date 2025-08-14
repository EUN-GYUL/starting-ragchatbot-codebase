import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        ai_gen = AIGenerator("test-key", "test-model")
        
        assert ai_gen.model == "test-model"
        assert ai_gen.base_params["model"] == "test-model"
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800
    
    def test_generate_response_no_tools(self, mock_ai_generator, mock_anthropic_client):
        """Test response generation without tools"""
        # Setup mock for direct response (no tools)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = mock_ai_generator.generate_response("What is AI?")
        
        # Verify API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == "test-model"
        assert call_args["messages"][0]["content"] == "What is AI?"
        assert "tools" not in call_args
        
        assert result == "Direct response without tools"
    
    def test_generate_response_with_conversation_history(self, mock_ai_generator, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with history")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        history = "Previous conversation context"
        result = mock_ai_generator.generate_response("New question", conversation_history=history)
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Check system prompt includes history
        assert history in call_args["system"]
        assert mock_ai_generator.SYSTEM_PROMPT in call_args["system"]
        
        assert result == "Response with history"
    
    def test_generate_response_with_tools_no_tool_use(self, mock_ai_generator, mock_anthropic_client, tool_manager):
        """Test response generation with tools available but not used"""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response without tool use")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = tool_manager.get_tool_definitions()
        result = mock_ai_generator.generate_response(
            "General question",
            tools=tools,
            tool_manager=tool_manager
        )
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Verify tools were provided
        assert "tools" in call_args
        assert call_args["tool_choice"] == {"type": "auto"}
        assert len(call_args["tools"]) == 1
        assert call_args["tools"][0]["name"] == "search_course_content"
        
        assert result == "Response without tool use"
    
    def test_generate_response_with_tool_use(self, mock_ai_generator, mock_anthropic_client):
        """Test response generation with tool use"""
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search course content"}
        ]
        mock_tool_manager.execute_tool.return_value = "Tool result content"
        
        # First response: tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.id = "tool_123"
        mock_content_block.input = {"query": "test search"}
        mock_tool_response.content = [mock_content_block]
        
        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Final response after tool use")]
        
        # Configure mock to return different responses on each call
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        tools = mock_tool_manager.get_tool_definitions()
        result = mock_ai_generator.generate_response(
            "Search for course content",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test search"
        )
        
        assert result == "Final response after tool use"
    
    def test_handle_tool_execution_single_tool(self):
        """Test handling of single tool execution"""
        # Create fresh mocks for this test
        mock_anthropic_client = Mock()
        ai_gen = AIGenerator("test-key", "test-model")
        ai_gen.client = mock_anthropic_client
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python is a programming language..."
        
        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.id = "tool_456"
        mock_content_block.input = {"query": "Python basics", "course_name": "Programming Course"}
        mock_initial_response.content = [mock_content_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Here's what I found about Python basics")]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Search for Python basics"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content"}]
        }
        
        result = ai_gen._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics",
            course_name="Programming Course"
        )
        
        # Verify final API call structure
        final_call_args = mock_anthropic_client.messages.create.call_args[1]
        messages = final_call_args["messages"]
        
        # Should have: user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Check tool result format
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_456"
        assert tool_results[0]["content"] == "Python is a programming language..."
        
        assert result == "Here's what I found about Python basics"
    
    def test_handle_tool_execution_multiple_tools(self):
        """Test handling of multiple tool executions in one response"""
        # Create fresh mocks for this test
        mock_anthropic_client = Mock()
        ai_gen = AIGenerator("test-key", "test-model")
        ai_gen.client = mock_anthropic_client
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Python content...",
            "JavaScript content..."
        ]
        
        # Mock initial response with multiple tool uses
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        mock_content_block_1 = Mock()
        mock_content_block_1.type = "tool_use"
        mock_content_block_1.name = "search_course_content"
        mock_content_block_1.id = "tool_1"
        mock_content_block_1.input = {"query": "Python basics"}
        
        mock_content_block_2 = Mock()
        mock_content_block_2.type = "tool_use"
        mock_content_block_2.name = "search_course_content"
        mock_content_block_2.id = "tool_2"
        mock_content_block_2.input = {"query": "JavaScript fundamentals"}
        
        mock_initial_response.content = [mock_content_block_1, mock_content_block_2]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Here's information about both languages")]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "test-model", 
            "messages": [{"role": "user", "content": "Compare Python and JavaScript"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content"}]
        }
        
        result = ai_gen._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Check tool result structure
        final_call_args = mock_anthropic_client.messages.create.call_args[1]
        tool_results = final_call_args["messages"][2]["content"]
        
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[1]["tool_use_id"] == "tool_2"
        
        assert result == "Here's information about both languages"
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
        assert "Sequential tool usage" in AIGenerator.SYSTEM_PROMPT
        assert "up to 2 rounds of tool calls" in AIGenerator.SYSTEM_PROMPT
        assert "tool_result" not in AIGenerator.SYSTEM_PROMPT.lower()  # No meta-commentary
        assert "Brief, Concise and focused" in AIGenerator.SYSTEM_PROMPT
    
    @patch('anthropic.Anthropic')
    def test_anthropic_client_initialization(self, mock_anthropic_class):
        """Test that Anthropic client is initialized correctly"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        ai_gen = AIGenerator("test-api-key", "test-model")
        
        mock_anthropic_class.assert_called_once_with(api_key="test-api-key")
        assert ai_gen.client == mock_client
    
    def test_api_parameters_structure(self, mock_ai_generator, mock_anthropic_client):
        """Test that API parameters are structured correctly"""
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        mock_ai_generator.generate_response(
            "Test query",
            conversation_history="Previous context",
            tools=[{"name": "test_tool"}]
        )
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Check required parameters
        assert "model" in call_args
        assert "messages" in call_args
        assert "system" in call_args
        assert "temperature" in call_args
        assert "max_tokens" in call_args
        assert "tools" in call_args
        assert "tool_choice" in call_args
        
        # Check parameter values
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["tool_choice"] == {"type": "auto"}
    
    def test_error_handling_missing_tool_manager(self, mock_ai_generator, mock_anthropic_client):
        """Test handling when tool_use response received but no tool_manager provided"""
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [Mock(text="Tool use attempt")]
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # This should not crash, just return the content directly
        result = mock_ai_generator.generate_response(
            "Test query",
            tools=[{"name": "test_tool"}],
            tool_manager=None
        )
        
        # Should return content directly since no tool_manager provided
        assert result == "Tool use attempt"
    
    def test_sequential_tool_calling_two_rounds(self):
        """Test sequential tool calling with two rounds"""
        # Create fresh mocks for this test
        mock_anthropic_client = Mock()
        ai_gen = AIGenerator("test-key", "test-model")
        ai_gen.client = mock_anthropic_client
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course found: Machine Learning Basics",
            "Related courses: Advanced ML, Deep Learning"
        ]
        
        # First response: first tool use
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_first_content_block = Mock()
        mock_first_content_block.type = "tool_use"
        mock_first_content_block.name = "get_course_outline"
        mock_first_content_block.id = "tool_round_1"
        mock_first_content_block.input = {"course_name": "ML Course"}
        mock_first_response.content = [mock_first_content_block]
        
        # Second response: second tool use
        mock_second_response = Mock()
        mock_second_response.stop_reason = "tool_use"
        mock_second_content_block = Mock()
        mock_second_content_block.type = "tool_use"
        mock_second_content_block.name = "search_course_content"
        mock_second_content_block.id = "tool_round_2"
        mock_second_content_block.input = {"query": "machine learning", "course_name": "ML Course"}
        mock_second_response.content = [mock_second_content_block]
        
        # Final response: answer after two rounds
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Based on both tool results, here's your answer")]
        
        # Configure mock to return responses in sequence
        # Note: the first response is already passed to _handle_tool_execution
        mock_anthropic_client.messages.create.side_effect = [
            mock_second_response, mock_final_response
        ]
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Find ML course and related content"}],
            "system": "Test system prompt",
            "tools": [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        }
        
        result = ai_gen._handle_tool_execution(
            mock_first_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify both tools were executed in sequence
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify three API calls were made (initial tool use, second tool use, final response)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify first tool call
        first_call = mock_tool_manager.execute_tool.call_args_list[0]
        assert first_call[0][0] == "get_course_outline"
        assert first_call[1]["course_name"] == "ML Course"
        
        # Verify second tool call
        second_call = mock_tool_manager.execute_tool.call_args_list[1]
        assert second_call[0][0] == "search_course_content"
        assert second_call[1]["query"] == "machine learning"
        
        # Verify final result
        assert result == "Based on both tool results, here's your answer"
    
    def test_sequential_tool_calling_max_rounds_exceeded(self):
        """Test that sequential tool calling stops at max rounds"""
        # Create fresh mocks for this test
        mock_anthropic_client = Mock()
        ai_gen = AIGenerator("test-key", "test-model")
        ai_gen.client = mock_anthropic_client
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First tool result",
            "Second tool result"
        ]
        
        # First response: tool use
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_first_content_block = Mock()
        mock_first_content_block.type = "tool_use"
        mock_first_content_block.name = "search_course_content"
        mock_first_content_block.id = "tool_1"
        mock_first_content_block.input = {"query": "first search"}
        mock_first_response.content = [mock_first_content_block]
        
        # Second response: tool use (should be the last round)
        mock_second_response = Mock()
        mock_second_response.stop_reason = "tool_use"
        mock_second_content_block = Mock()
        mock_second_content_block.type = "tool_use"
        mock_second_content_block.name = "search_course_content"
        mock_second_content_block.id = "tool_2"
        mock_second_content_block.input = {"query": "second search"}
        mock_second_response.content = [mock_second_content_block]
        
        # Third response: would be tool use but should not happen due to max rounds
        mock_third_response = Mock()
        mock_third_response.stop_reason = "tool_use"
        mock_third_response.content = [Mock(text="Should not reach here")]
        
        # Configure mock to return responses - only first two should be called
        mock_anthropic_client.messages.create.side_effect = [
            mock_second_response, mock_third_response
        ]
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content"}]
        }
        
        result = ai_gen._handle_tool_execution(
            mock_first_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify only 2 tools were executed (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify only 2 API calls were made (not 3)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Should return the last response content even if it wanted more tools
        assert "Should not reach here" in result
    
    def test_sequential_tool_calling_with_error_in_second_round(self):
        """Test sequential tool calling with error in second round"""
        # Create fresh mocks for this test
        mock_anthropic_client = Mock()
        ai_gen = AIGenerator("test-key", "test-model")
        ai_gen.client = mock_anthropic_client
        
        # Create mock tool manager with error on second call
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First tool result",
            Exception("Tool execution failed")
        ]
        
        # First response: tool use
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_first_content_block = Mock()
        mock_first_content_block.type = "tool_use"
        mock_first_content_block.name = "search_course_content"
        mock_first_content_block.id = "tool_1"
        mock_first_content_block.input = {"query": "first search"}
        mock_first_response.content = [mock_first_content_block]
        
        # Second response: tool use
        mock_second_response = Mock()
        mock_second_response.stop_reason = "tool_use"
        mock_second_content_block = Mock()
        mock_second_content_block.type = "tool_use"
        mock_second_content_block.name = "search_course_content"
        mock_second_content_block.id = "tool_2"
        mock_second_content_block.input = {"query": "second search"}
        mock_second_response.content = [mock_second_content_block]
        
        # Final response: answer with error handling
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Response with error handled")]
        
        mock_anthropic_client.messages.create.side_effect = [
            mock_second_response, mock_final_response
        ]
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content"}]
        }
        
        result = ai_gen._handle_tool_execution(
            mock_first_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify both tools were attempted
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify that the error was handled gracefully
        assert result == "Response with error handled"
    
    def test_sequential_tool_calling_early_termination(self):
        """Test sequential tool calling that terminates early when no more tools needed"""
        # Create fresh mocks for this test
        mock_anthropic_client = Mock()
        ai_gen = AIGenerator("test-key", "test-model")
        ai_gen.client = mock_anthropic_client
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # First response: tool use
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_first_content_block = Mock()
        mock_first_content_block.type = "tool_use"
        mock_first_content_block.name = "search_course_content"
        mock_first_content_block.id = "tool_1"
        mock_first_content_block.input = {"query": "search"}
        mock_first_response.content = [mock_first_content_block]
        
        # Second response: no more tool use, direct answer
        mock_second_response = Mock()
        mock_second_response.stop_reason = "end_turn"
        mock_second_response.content = [Mock(text="Final answer after one tool use")]
        
        mock_anthropic_client.messages.create.side_effect = [mock_second_response]
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test query"}],
            "system": "Test system prompt",
            "tools": [{"name": "search_course_content"}]
        }
        
        result = ai_gen._handle_tool_execution(
            mock_first_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify only one tool was executed
        assert mock_tool_manager.execute_tool.call_count == 1
        
        # Verify only one API call was made (since second response was end_turn)
        assert mock_anthropic_client.messages.create.call_count == 1
        
        # Verify correct termination
        assert result == "Final answer after one tool use"