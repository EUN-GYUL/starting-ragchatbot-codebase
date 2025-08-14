import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystemInitialization:
    """Test RAG system initialization and component setup"""
    
    def test_init_components(self, test_config):
        """Test that all components are properly initialized"""
        rag = RAGSystem(test_config)
        
        # Verify all components exist
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None
        
        # Verify tools are registered
        assert "search_course_content" in rag.tool_manager.tools
        assert "get_course_outline" in rag.tool_manager.tools
    
    def test_config_propagation(self, test_config):
        """Test that configuration is properly propagated to components"""
        rag = RAGSystem(test_config)
        
        # Check document processor config
        assert rag.document_processor.chunk_size == test_config.CHUNK_SIZE
        assert rag.document_processor.chunk_overlap == test_config.CHUNK_OVERLAP
        
        # Check vector store config
        assert rag.vector_store.max_results == test_config.MAX_RESULTS
        
        # Check session manager config
        assert rag.session_manager.max_history == test_config.MAX_HISTORY


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""
    
    def test_add_course_document_success(self, test_config, temp_dir):
        """Test successful course document addition"""
        # Create a test document
        test_file = os.path.join(temp_dir, "test_course.txt")
        with open(test_file, 'w') as f:
            f.write("""Course Title: Test Course
Course Link: http://example.com
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: http://example.com/lesson1
This is the content of lesson 1.

Lesson 2: Advanced Topics
Lesson Link: http://example.com/lesson2
This is the content of lesson 2.""")
        
        rag = RAGSystem(test_config)
        course, chunk_count = rag.add_course_document(test_file)
        
        assert course is not None
        assert course.title == "Test Course"
        assert course.instructor == "Test Instructor"
        assert len(course.lessons) == 2
        assert chunk_count > 0
    
    def test_add_course_document_file_not_found(self, test_config):
        """Test handling of non-existent files"""
        rag = RAGSystem(test_config)
        course, chunk_count = rag.add_course_document("/nonexistent/file.txt")
        
        assert course is None
        assert chunk_count == 0
    
    def test_add_course_folder_success(self, test_config, temp_dir):
        """Test adding multiple courses from folder"""
        # Create test documents
        for i in range(2):
            test_file = os.path.join(temp_dir, f"course{i+1}.txt")
            with open(test_file, 'w') as f:
                f.write(f"""Course Title: Test Course {i+1}
Course Link: http://example.com/course{i+1}
Course Instructor: Instructor {i+1}

Lesson 1: Introduction
Lesson Link: http://example.com/course{i+1}/lesson1
This is the content of course {i+1} lesson 1.""")
        
        rag = RAGSystem(test_config)
        total_courses, total_chunks = rag.add_course_folder(temp_dir)
        
        assert total_courses == 2
        assert total_chunks > 0
    
    def test_add_course_folder_nonexistent(self, test_config):
        """Test handling of non-existent folder"""
        rag = RAGSystem(test_config)
        total_courses, total_chunks = rag.add_course_folder("/nonexistent/folder")
        
        assert total_courses == 0
        assert total_chunks == 0
    
    def test_add_course_folder_clear_existing(self, test_config, temp_dir):
        """Test clearing existing data before adding new courses"""
        # Create test document
        test_file = os.path.join(temp_dir, "course1.txt")
        with open(test_file, 'w') as f:
            f.write("""Course Title: Test Course
Course Link: http://example.com
Course Instructor: Test Instructor

Lesson 1: Introduction
This is the content.""")
        
        rag = RAGSystem(test_config)
        
        # Add first time
        rag.add_course_folder(temp_dir)
        initial_count = rag.vector_store.get_course_count()
        
        # Add again with clear_existing=True
        total_courses, total_chunks = rag.add_course_folder(temp_dir, clear_existing=True)
        final_count = rag.vector_store.get_course_count()
        
        assert total_courses == 1
        assert final_count == initial_count  # Should be same after clearing and re-adding


class TestRAGSystemQuerying:
    """Test query processing functionality"""
    
    @patch('ai_generator.AIGenerator.generate_response')
    def test_query_without_session(self, mock_generate, test_config):
        """Test query processing without session ID"""
        mock_generate.return_value = "Test AI response"
        
        rag = RAGSystem(test_config)
        response, sources = rag.query("What is machine learning?")
        
        assert response == "Test AI response"
        assert isinstance(sources, list)
        
        # Verify AI generator was called correctly
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        
        assert "What is machine learning?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
    
    @patch('ai_generator.AIGenerator.generate_response')
    def test_query_with_session(self, mock_generate, test_config):
        """Test query processing with session ID"""
        mock_generate.return_value = "Test AI response with context"
        
        rag = RAGSystem(test_config)
        
        # Create a session and add some history
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "Previous question", "Previous answer")
        
        response, sources = rag.query("Follow-up question", session_id=session_id)
        
        assert response == "Test AI response with context"
        
        # Verify conversation history was passed
        call_args = mock_generate.call_args
        assert call_args[1]["conversation_history"] is not None
    
    @patch('ai_generator.AIGenerator.generate_response')
    def test_query_updates_session_history(self, mock_generate, test_config):
        """Test that queries update session history"""
        mock_generate.return_value = "AI response"
        
        rag = RAGSystem(test_config)
        session_id = rag.session_manager.create_session()
        
        # Initial query
        rag.query("First question", session_id=session_id)
        
        # Check history was updated
        history = rag.session_manager.get_conversation_history(session_id)
        assert "First question" in history
        assert "AI response" in history
    
    @patch('ai_generator.AIGenerator.generate_response') 
    def test_query_tool_sources_tracking(self, mock_generate, test_config):
        """Test that sources from tools are properly tracked"""
        mock_generate.return_value = "AI response using tools"
        
        rag = RAGSystem(test_config)
        
        # Mock the tool manager to return sources
        rag.tool_manager.get_last_sources = Mock(return_value=["Source 1", "Source 2"])
        rag.tool_manager.reset_sources = Mock()
        
        response, sources = rag.query("Search for Python basics")
        
        assert sources == ["Source 1", "Source 2"]
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()
    
    def test_query_integration_with_real_tools(self, test_config, temp_dir):
        """Test query integration with real search tools"""
        # Create a test document
        test_file = os.path.join(temp_dir, "programming_course.txt")
        with open(test_file, 'w') as f:
            f.write("""Course Title: Programming Fundamentals
Course Link: http://example.com/programming
Course Instructor: Jane Doe

Lesson 1: Python Basics
Lesson Link: http://example.com/programming/lesson1
Python is a high-level programming language. It is widely used for web development, data analysis, and machine learning.

Lesson 2: Data Structures
Lesson Link: http://example.com/programming/lesson2
Data structures are ways of organizing data. Common types include lists, dictionaries, and sets.""")
        
        rag = RAGSystem(test_config)
        rag.add_course_document(test_file)
        
        # Mock AI generator to simulate tool use
        with patch.object(rag.ai_generator, 'generate_response') as mock_generate:
            # Simulate AI deciding to use search tool
            def mock_ai_response(query, conversation_history=None, tools=None, tool_manager=None):
                if tool_manager:
                    # Simulate tool execution
                    search_result = tool_manager.execute_tool(
                        "search_course_content",
                        query="Python basics"
                    )
                    return f"Based on the course materials: {search_result[:100]}..."
                return "No tools available"
            
            mock_generate.side_effect = mock_ai_response
            
            response, sources = rag.query("Tell me about Python")
            
            assert "Python is a high-level programming language" in response
            assert len(sources) > 0  # Should have sources from search


class TestRAGSystemAnalytics:
    """Test analytics and reporting functionality"""
    
    def test_get_course_analytics_empty(self, test_config):
        """Test analytics with no courses"""
        rag = RAGSystem(test_config)
        analytics = rag.get_course_analytics()
        
        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []
    
    def test_get_course_analytics_with_courses(self, test_config, temp_dir):
        """Test analytics with courses loaded"""
        # Create test documents
        test_file = os.path.join(temp_dir, "test_course.txt")
        with open(test_file, 'w') as f:
            f.write("""Course Title: Analytics Test Course
Course Link: http://example.com
Course Instructor: Test Instructor

Lesson 1: Introduction
Content here.""")
        
        rag = RAGSystem(test_config)
        rag.add_course_document(test_file)
        
        analytics = rag.get_course_analytics()
        
        assert analytics["total_courses"] == 1
        assert "Analytics Test Course" in analytics["course_titles"]


class TestRAGSystemErrorHandling:
    """Test error handling in various scenarios"""
    
    @patch('ai_generator.AIGenerator.generate_response')
    def test_query_ai_generator_exception(self, mock_generate, test_config):
        """Test handling of AI generator exceptions"""
        mock_generate.side_effect = Exception("API Error")
        
        rag = RAGSystem(test_config)
        
        # This should not crash the system
        with pytest.raises(Exception):
            rag.query("Test question")
    
    def test_query_with_invalid_session_id(self, test_config):
        """Test query with non-existent session ID"""
        rag = RAGSystem(test_config)
        
        # Should handle gracefully - create new session or return empty history
        response, sources = rag.query("Test question", session_id="invalid-session-id")
        
        # Should not crash
        assert isinstance(response, str)
        assert isinstance(sources, list)
    
    def test_document_processing_error_handling(self, test_config, temp_dir):
        """Test error handling during document processing"""
        # Create invalid document
        test_file = os.path.join(temp_dir, "invalid.txt")
        with open(test_file, 'w') as f:
            f.write("Invalid document format without proper headers")
        
        rag = RAGSystem(test_config)
        course, chunk_count = rag.add_course_document(test_file)
        
        # Should handle gracefully
        assert course is None
        assert chunk_count == 0


class TestRAGSystemPerformance:
    """Test performance-related aspects"""
    
    def test_session_memory_management(self, test_config):
        """Test that session history is properly limited"""
        rag = RAGSystem(test_config)
        session_id = rag.session_manager.create_session()
        
        # Add more exchanges than MAX_HISTORY allows
        for i in range(test_config.MAX_HISTORY + 2):
            rag.session_manager.add_exchange(
                session_id,
                f"Question {i}",
                f"Answer {i}"
            )
        
        history = rag.session_manager.get_conversation_history(session_id)
        
        # History should be limited
        # Each exchange creates 2 lines (Q and A), so MAX_HISTORY * 2 lines total
        line_count = len([line for line in history.split('\n') if line.strip()])
        assert line_count <= test_config.MAX_HISTORY * 2 + 2  # Allow some flexibility for formatting
    
    def test_vector_store_limits(self, test_config, temp_dir):
        """Test that vector store respects MAX_RESULTS limit"""
        # Create course with multiple lessons to generate multiple chunks
        test_file = os.path.join(temp_dir, "large_course.txt")
        with open(test_file, 'w') as f:
            f.write("""Course Title: Large Course
Course Link: http://example.com
Course Instructor: Test Instructor

""")
            # Add multiple lessons with similar content to get multiple matches
            for i in range(10):
                f.write(f"""Lesson {i+1}: Python Programming {i+1}
Lesson Link: http://example.com/lesson{i+1}
Python is a programming language used for development. This lesson covers Python fundamentals and basic concepts.

""")
        
        rag = RAGSystem(test_config)
        rag.add_course_document(test_file)
        
        # Search should respect MAX_RESULTS limit
        results = rag.vector_store.search("Python programming")
        
        assert len(results.documents) <= test_config.MAX_RESULTS