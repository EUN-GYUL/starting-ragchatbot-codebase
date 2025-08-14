import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestVectorStore:
    """Test suite for VectorStore functionality"""

    def test_max_results_zero_bug(self, temp_dir):
        """Test that MAX_RESULTS=0 causes no results to be returned"""
        # Create vector store with MAX_RESULTS=0 to reproduce the bug
        store = VectorStore(
            chroma_path=os.path.join(temp_dir, "test_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,  # This should cause the bug
        )

        # Add some test data
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://example.com",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Test Lesson",
                    lesson_link="http://example.com/1",
                )
            ],
        )
        chunks = [
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
                content="This is test content that should be searchable",
            )
        ]

        store.add_course_metadata(course)
        store.add_course_content(chunks)

        # Search should return no results due to MAX_RESULTS=0
        results = store.search("test content")

        # This test should pass when the bug exists (empty results)
        # and fail when the bug is fixed
        assert (
            results.is_empty()
        ), "MAX_RESULTS=0 should return no results (reproducing bug)"

    def test_max_results_fixed(self, temp_dir):
        """Test that MAX_RESULTS=5 returns proper results"""
        # Create vector store with MAX_RESULTS=5 (fixed value)
        store = VectorStore(
            chroma_path=os.path.join(temp_dir, "test_chroma_fixed"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,  # Fixed value
        )

        # Add some test data
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://example.com",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Test Lesson",
                    lesson_link="http://example.com/1",
                )
            ],
        )
        chunks = [
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
                content="This is test content that should be searchable",
            )
        ]

        store.add_course_metadata(course)
        store.add_course_content(chunks)

        # Search should return results with MAX_RESULTS=5
        results = store.search("test content")

        # This should return results when MAX_RESULTS > 0
        assert not results.is_empty(), "MAX_RESULTS=5 should return results"
        assert len(results.documents) > 0, "Should have at least one document"
        assert results.error is None, "Should not have errors"

    def test_search_with_course_filter(self, real_vector_store):
        """Test search with course name filtering"""
        results = real_vector_store.search("introduction", course_name="Test Course")

        assert not results.is_empty(), "Should find results in specified course"
        assert results.error is None, "Should not have errors"

        # Check that all results are from the specified course
        for metadata in results.metadata:
            assert metadata["course_title"] == "Test Course"

    def test_search_with_lesson_filter(self, real_vector_store):
        """Test search with lesson number filtering"""
        results = real_vector_store.search("introduction", lesson_number=1)

        assert not results.is_empty(), "Should find results in specified lesson"
        assert results.error is None, "Should not have errors"

        # Check that all results are from the specified lesson
        for metadata in results.metadata:
            assert metadata["lesson_number"] == 1

    def test_search_with_both_filters(self, real_vector_store):
        """Test search with both course and lesson filters"""
        results = real_vector_store.search(
            "introduction", course_name="Test Course", lesson_number=1
        )

        assert not results.is_empty(), "Should find results with both filters"
        assert results.error is None, "Should not have errors"

        # Check that all results match both filters
        for metadata in results.metadata:
            assert metadata["course_title"] == "Test Course"
            assert metadata["lesson_number"] == 1

    def test_search_nonexistent_course(self, real_vector_store):
        """Test search with non-existent course name"""
        results = real_vector_store.search(
            "introduction", course_name="Nonexistent Course"
        )

        assert results.error is not None, "Should return error for nonexistent course"
        assert "No course found" in results.error

    def test_resolve_course_name(self, real_vector_store):
        """Test course name resolution"""
        # Should find exact match
        resolved = real_vector_store._resolve_course_name("Test Course")
        assert resolved == "Test Course"

        # Should find partial match
        resolved = real_vector_store._resolve_course_name("Test")
        assert resolved == "Test Course"

    def test_build_filter_no_params(self, real_vector_store):
        """Test filter building with no parameters"""
        filter_dict = real_vector_store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, real_vector_store):
        """Test filter building with course only"""
        filter_dict = real_vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, real_vector_store):
        """Test filter building with lesson only"""
        filter_dict = real_vector_store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_both(self, real_vector_store):
        """Test filter building with both course and lesson"""
        filter_dict = real_vector_store._build_filter("Test Course", 1)
        expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 1}]}
        assert filter_dict == expected

    def test_add_course_metadata(self, real_vector_store, sample_course):
        """Test adding course metadata"""
        # Course should already be added by fixture, check it exists
        titles = real_vector_store.get_existing_course_titles()
        assert sample_course.title in titles

    def test_add_course_content(self, real_vector_store, sample_chunks):
        """Test adding course content chunks"""
        # Chunks should already be added by fixture, test search works
        results = real_vector_store.search("introduction")
        assert not results.is_empty()

    def test_get_course_count(self, real_vector_store):
        """Test getting course count"""
        count = real_vector_store.get_course_count()
        assert count > 0, "Should have at least one course"

    def test_get_lesson_link(self, real_vector_store):
        """Test getting lesson link"""
        link = real_vector_store.get_lesson_link("Test Course", 1)
        assert link == "http://example.com/lesson1"

        # Test non-existent lesson
        link = real_vector_store.get_lesson_link("Test Course", 999)
        assert link is None

    @patch("chromadb.PersistentClient")
    def test_search_exception_handling(self, mock_client_class, temp_dir):
        """Test that search exceptions are handled properly"""
        # Mock client to raise exception
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Database error")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = VectorStore(
            chroma_path=os.path.join(temp_dir, "test_error"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

        results = store.search("test query")
        assert results.error is not None
        assert "Search error" in results.error


class TestSearchResults:
    """Test suite for SearchResults class"""

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)
        assert results.is_empty()
        assert results.error is None

    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB results with data"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)
        assert not results.is_empty()
        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Test error")
        assert results.is_empty()
        assert results.error == "Test error"

    def test_is_empty(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()

        # Non-empty results
        results = SearchResults(["doc"], [{}], [0.1])
        assert not results.is_empty()
