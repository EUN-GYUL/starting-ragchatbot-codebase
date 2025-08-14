import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


@dataclass
class TestConfig:
    """Test configuration with safe defaults"""
    ANTHROPIC_API_KEY: str = "test-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = ""  # Will be set in fixture


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration with temp directory"""
    config = TestConfig()
    config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma")
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction", lesson_link="http://example.com/lesson1"),
        Lesson(lesson_number=2, title="Advanced Topics", lesson_link="http://example.com/lesson2")
    ]
    return Course(
        title="Test Course",
        instructor="Test Instructor", 
        course_link="http://example.com/course",
        lessons=lessons
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
            content="This is the introduction to the test course. It covers basic concepts."
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1,
            content="More detailed information about the fundamentals and key principles."
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2,
            content="Advanced topics include complex algorithms and data structures."
        )
    ]


@pytest.fixture
def mock_vector_store(sample_chunks):
    """Create a mock vector store with predictable responses"""
    mock_store = Mock(spec=VectorStore)
    
    # Mock successful search results
    mock_store.search.return_value = SearchResults(
        documents=[chunk.content for chunk in sample_chunks],
        metadata=[{
            'course_title': chunk.course_title,
            'lesson_number': chunk.lesson_number,
            'chunk_index': chunk.chunk_index
        } for chunk in sample_chunks],
        distances=[0.1, 0.2, 0.3]
    )
    
    # Mock course resolution
    mock_store._resolve_course_name.return_value = "Test Course"
    mock_store.get_lesson_link.return_value = "http://example.com/lesson1"
    
    return mock_store


@pytest.fixture
def real_vector_store(test_config, sample_course, sample_chunks):
    """Create a real vector store with test data"""
    store = VectorStore(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
    
    # Add test data
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_chunks)
    
    return store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = Mock()
    
    # Don't pre-configure the client - let individual tests set their own expectations
    # This allows tests to have full control over mock behavior
    
    return mock_client


@pytest.fixture
def mock_ai_generator(mock_anthropic_client):
    """Create AI generator with mocked client"""
    ai_gen = AIGenerator("test-key", "test-model")
    ai_gen.client = mock_anthropic_client
    return ai_gen


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager


# Empty search results for testing failure cases
@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


# Error search results for testing error cases
@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults.empty("Test error message")