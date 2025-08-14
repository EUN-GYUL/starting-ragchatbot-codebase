import os
import sys
from unittest.mock import MagicMock, Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool functionality"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is properly formatted"""
        definition = course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_basic_query(
        self, course_search_tool, mock_vector_store, sample_chunks
    ):
        """Test basic query execution without filters"""
        # Setup mock to return sample data
        mock_vector_store.search.return_value = SearchResults(
            documents=[chunk.content for chunk in sample_chunks],
            metadata=[
                {
                    "course_title": chunk.course_title,
                    "lesson_number": chunk.lesson_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in sample_chunks
            ],
            distances=[0.1, 0.2, 0.3],
        )

        result = course_search_tool.execute("test query")

        # Verify search was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=None
        )

        # Verify result format
        assert isinstance(result, str)
        assert "Test Course" in result
        assert "introduction" in result.lower()
        assert "[Test Course - Lesson 1]" in result

    def test_execute_with_course_filter(self, course_search_tool, mock_vector_store):
        """Test query execution with course name filter"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )

        result = course_search_tool.execute("test query", course_name="Test Course")

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Test Course", lesson_number=None
        )

        assert "Test Course" in result

    def test_execute_with_lesson_filter(self, course_search_tool, mock_vector_store):
        """Test query execution with lesson number filter"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 2}],
            distances=[0.1],
        )

        result = course_search_tool.execute("test query", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=2
        )

        assert "Lesson 2" in result

    def test_execute_with_both_filters(self, course_search_tool, mock_vector_store):
        """Test query execution with both course and lesson filters"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )

        result = course_search_tool.execute(
            "test query", course_name="Test Course", lesson_number=1
        )

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Test Course", lesson_number=1
        )

        assert "Test Course" in result
        assert "Lesson 1" in result

    def test_execute_empty_results(self, course_search_tool, mock_vector_store):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = course_search_tool.execute("nonexistent query")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(
        self, course_search_tool, mock_vector_store
    ):
        """Test handling of empty results with filters applied"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[]
        )

        result = course_search_tool.execute(
            "test query", course_name="Test Course", lesson_number=1
        )

        assert "No relevant content found in course 'Test Course' in lesson 1" in result

    def test_execute_error_handling(self, course_search_tool, mock_vector_store):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = SearchResults.empty(
            "Database connection failed"
        )

        result = course_search_tool.execute("test query")

        assert result == "Database connection failed"

    def test_format_results_basic(self, course_search_tool):
        """Test basic result formatting"""
        results = SearchResults(
            documents=["Content from lesson 1", "Content from lesson 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Test Course", "lesson_number": 2, "chunk_index": 1},
            ],
            distances=[0.1, 0.2],
        )

        formatted = course_search_tool._format_results(results)

        assert "[Test Course - Lesson 1]" in formatted
        assert "[Test Course - Lesson 2]" in formatted
        assert "Content from lesson 1" in formatted
        assert "Content from lesson 2" in formatted

    def test_format_results_with_links(self, course_search_tool, mock_vector_store):
        """Test result formatting with lesson links"""
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )

        formatted = course_search_tool._format_results(results)

        # Verify lesson link was requested
        mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 1)

        # Verify sources are tracked
        assert len(course_search_tool.last_sources) == 1
        source = course_search_tool.last_sources[0]
        assert isinstance(source, dict)
        assert source["text"] == "Test Course - Lesson 1"
        assert source["link"] == "http://example.com/lesson1"

    def test_format_results_no_lesson_number(self, course_search_tool):
        """Test formatting results without lesson numbers"""
        results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "Test Course", "chunk_index": 0}],
            distances=[0.1],
        )

        formatted = course_search_tool._format_results(results)

        assert "[Test Course]" in formatted
        assert "Course overview content" in formatted

        # Should store plain text source
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0] == "Test Course"


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool functionality"""

    def test_get_tool_definition(self):
        """Test that outline tool definition is properly formatted"""
        mock_store = Mock()
        tool = CourseOutlineTool(mock_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["course_name"]
        assert "course_name" in definition["input_schema"]["properties"]

    def test_execute_existing_course(self, mock_vector_store):
        """Test outline retrieval for existing course"""
        # Setup mock responses
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "Test Course",
                    "instructor": "Test Instructor",
                    "course_link": "http://example.com/course",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Advanced Topics"}]',
                }
            ]
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Test Course")

        assert "**Course:** Test Course" in result
        assert "**Instructor:** Test Instructor" in result
        assert "**Course Link:** http://example.com/course" in result
        assert "1. Introduction" in result
        assert "2. Advanced Topics" in result

    def test_execute_nonexistent_course(self, mock_vector_store):
        """Test outline retrieval for non-existent course"""
        mock_vector_store._resolve_course_name.return_value = None

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Nonexistent Course")

        assert "No course found matching 'Nonexistent Course'" in result

    def test_execute_missing_metadata(self, mock_vector_store):
        """Test handling of missing course metadata"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.return_value = {"metadatas": []}

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Test Course")

        assert "Course metadata not found" in result

    def test_execute_exception_handling(self, mock_vector_store):
        """Test exception handling in outline tool"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.course_catalog.get.side_effect = Exception("Database error")

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Test Course")

        assert "Error retrieving course outline" in result


class TestToolManager:
    """Test suite for ToolManager functionality"""

    def test_register_tool(self, course_search_tool):
        """Test tool registration"""
        manager = ToolManager()
        manager.register_tool(course_search_tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == course_search_tool

    def test_register_tool_without_name(self):
        """Test registering tool without name raises error"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {}  # No name

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions"""
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, tool_manager, mock_vector_store):
        """Test tool execution"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
        )

        result = tool_manager.execute_tool("search_course_content", query="test")

        assert isinstance(result, str)
        assert "test content" in result

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, tool_manager, course_search_tool):
        """Test getting last sources from tools"""
        # Set up sources on the search tool
        course_search_tool.last_sources = ["source1", "source2"]

        sources = tool_manager.get_last_sources()
        assert sources == ["source1", "source2"]

    def test_get_last_sources_empty(self, tool_manager):
        """Test getting last sources when no sources exist"""
        sources = tool_manager.get_last_sources()
        assert sources == []

    def test_reset_sources(self, tool_manager, course_search_tool):
        """Test resetting sources from all tools"""
        # Set up sources on the search tool
        course_search_tool.last_sources = ["source1", "source2"]

        tool_manager.reset_sources()

        assert course_search_tool.last_sources == []
