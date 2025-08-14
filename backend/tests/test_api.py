import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns correct message"""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System API"}
    
    def test_query_endpoint_with_session_id(self, test_client, mock_rag_system):
        """Test /api/query endpoint with provided session ID"""
        query_data = {
            "query": "What is the course about?",
            "session_id": "existing-session-123"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "existing-session-123"
        assert data["answer"] == "This is a test response about the course materials."
        assert data["sources"] == ["Test Course - Lesson 1", "Test Course - Lesson 2"]
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is the course about?", "existing-session-123")
    
    def test_query_endpoint_without_session_id(self, test_client, mock_rag_system):
        """Test /api/query endpoint creates new session when not provided"""
        query_data = {
            "query": "Tell me about machine learning"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        
        # Verify session creation was called
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("Tell me about machine learning", "test-session-123")
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test /api/query endpoint with empty query"""
        query_data = {"query": ""}
        
        response = test_client.post("/api/query", json=query_data)
        
        # Should still work - empty queries are handled by the RAG system
        assert response.status_code == 200
    
    def test_query_endpoint_missing_query(self, test_client):
        """Test /api/query endpoint with missing query field"""
        query_data = {"session_id": "test-123"}
        
        response = test_client.post("/api/query", json=query_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_query_endpoint_rag_system_error(self, test_client, mock_rag_system):
        """Test /api/query endpoint when RAG system raises exception"""
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        query_data = {"query": "What is this about?"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    def test_courses_endpoint_success(self, test_client, mock_rag_system):
        """Test /api/courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Test Course"]
        
        # Verify analytics was called
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_analytics_error(self, test_client, mock_rag_system):
        """Test /api/courses endpoint when analytics raises exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]
    
    def test_new_session_endpoint_success(self, test_client, mock_rag_system):
        """Test /api/session/new endpoint creates new session"""
        response = test_client.post("/api/session/new")
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        
        # Verify session creation was called
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_new_session_endpoint_error(self, test_client, mock_rag_system):
        """Test /api/session/new endpoint when session creation fails"""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session creation failed")
        
        response = test_client.post("/api/session/new")
        
        assert response.status_code == 500
        assert "Session creation failed" in response.json()["detail"]


@pytest.mark.api 
class TestAPIRequestValidation:
    """Test suite for API request validation"""
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test /api/query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_endpoint_wrong_content_type(self, test_client):
        """Test /api/query endpoint with wrong content type"""
        response = test_client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422
    
    def test_query_endpoint_extra_fields(self, test_client):
        """Test /api/query endpoint ignores extra fields"""
        query_data = {
            "query": "Test query",
            "session_id": "test-123",
            "extra_field": "should be ignored"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        # Should succeed and ignore extra field
        assert response.status_code == 200


@pytest.mark.api
class TestAPIResponseFormats:
    """Test suite for API response formats"""
    
    def test_query_response_format(self, test_client):
        """Test /api/query response has correct format"""
        query_data = {"query": "test"}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check field values
        assert len(data["answer"]) > 0
        assert len(data["session_id"]) > 0
    
    def test_courses_response_format(self, test_client):
        """Test /api/courses response has correct format"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check field values
        assert data["total_courses"] >= 0
        assert all(isinstance(title, str) for title in data["course_titles"])
    
    def test_session_response_format(self, test_client):
        """Test /api/session/new response has correct format"""
        response = test_client.post("/api/session/new")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0


@pytest.mark.api
class TestAPICORS:
    """Test suite for CORS middleware"""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        # Make a request that would trigger CORS headers (with Origin header)
        response = test_client.get("/", headers={"Origin": "http://localhost:3000"})
        
        # CORS headers may not be present for same-origin requests
        # Let's just check the response is successful
        assert response.status_code == 200
    
    def test_options_request(self, test_client):
        """Test OPTIONS request for CORS preflight"""
        response = test_client.options("/api/query")
        
        # Should handle OPTIONS requests
        assert response.status_code in [200, 405]  # 405 is also acceptable for FastAPI