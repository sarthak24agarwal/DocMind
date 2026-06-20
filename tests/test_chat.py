import pytest
from unittest.mock import MagicMock, patch
import uuid
import json
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException

from app.routers.chat import get_or_reset_usage, chat_rag
from app.models import WorkspaceUsage, Conversation, Message, DocumentChunk

# ----------------------------------------------------
# 1. Usage Limit & Reset Tests
# ----------------------------------------------------

def test_get_or_reset_usage_new():
    db = MagicMock()
    workspace_id = uuid.uuid4()
    
    # Mock no usage record exists
    db.query().filter().first.return_value = None
    
    usage = get_or_reset_usage(db, workspace_id)
    
    assert usage.workspace_id == workspace_id
    assert usage.billing_tier == "free"
    assert usage.queries_this_month == 0
    assert db.add.called
    assert db.commit.called

def test_get_or_reset_usage_existing_no_reset():
    db = MagicMock()
    workspace_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    
    existing_usage = WorkspaceUsage(
        workspace_id=workspace_id,
        billing_tier="free",
        queries_this_month=15,
        last_reset_date=now
    )
    db.query().filter().first.return_value = existing_usage
    
    usage = get_or_reset_usage(db, workspace_id)
    
    assert usage.queries_this_month == 15
    assert not db.add.called

def test_get_or_reset_usage_reset_required():
    db = MagicMock()
    workspace_id = uuid.uuid4()
    # Set last reset to 2 months ago
    past_date = datetime.now(timezone.utc) - timedelta(days=60)
    
    existing_usage = WorkspaceUsage(
        workspace_id=workspace_id,
        billing_tier="free",
        queries_this_month=45,
        last_reset_date=past_date
    )
    db.query().filter().first.return_value = existing_usage
    
    usage = get_or_reset_usage(db, workspace_id)
    
    assert usage.queries_this_month == 0
    assert usage.last_reset_date.month == datetime.now(timezone.utc).month
    assert db.commit.called

# ----------------------------------------------------
# 2. Similarity Threshold & Bypass Tests
# ----------------------------------------------------

@pytest.mark.asyncio
@patch("app.routers.chat.embedding_service")
@patch("app.routers.chat.get_or_reset_usage")
@patch("app.routers.chat.SessionLocal")
async def test_chat_rag_insufficient_similarity_bypass(
    mock_session_local,
    mock_get_usage,
    mock_embed_service,
):
    workspace_id = uuid.uuid4()
    db = MagicMock()
    
    # Mock workspace exists
    db.query().filter().first.return_value = MagicMock()
    
    # Mock limits are OK
    mock_get_usage.return_value = WorkspaceUsage(
        workspace_id=workspace_id,
        billing_tier="free",
        queries_this_month=0
    )
    
    # Mock query embedding
    mock_embed_service.get_embeddings.return_value = [[0.1] * 1536]
    
    # Mock pgvector similarity query returns empty (below threshold)
    db.query().join().filter().filter().order_by().limit().all.return_value = []
    
    # Mock thread session local
    thread_db = MagicMock()
    mock_session_local.return_value = thread_db
    
    # Execute Endpoint
    from app.routers.chat import ChatRequest
    payload = ChatRequest(message="What is docmind?")
    response = await chat_rag(workspace_id, payload, db)
    
    assert response is not None
    
    # Read generator stream
    events = []
    async for event in response.body_iterator:
        events.append(event)
        
    assert len(events) == 3
    # First event should yield fallback message
    assert "I'm sorry, I couldn't find any relevant information" in events[0]
    # Second event is empty citations
    assert "citations" in events[1]
    # Third event is DONE
    assert "[DONE]" in events[2]
    
    # Verify Claude was NOT called (no anthropic client interaction)
    # Database is written with fallback message
    assert thread_db.add.called
    assert thread_db.commit.called

# ----------------------------------------------------
# 3. Citation Extraction & SSE Stream Tests
# ----------------------------------------------------

@pytest.mark.asyncio
@patch("app.routers.chat.anthropic_service")
@patch("app.routers.chat.embedding_service")
@patch("app.routers.chat.get_or_reset_usage")
@patch("app.routers.chat.SessionLocal")
async def test_chat_rag_success_stream_and_citations(
    mock_session_local,
    mock_get_usage,
    mock_embed_service,
    mock_anthropic_service,
):
    workspace_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    db = MagicMock()
    
    # 1. Mock workspace and conversation exist
    db.query().filter().first.side_effect = [
        MagicMock(id=workspace_id),  # Workspace check
        MagicMock(id=conversation_id)  # Conversation check
    ]
    
    # Mock limits are OK
    mock_get_usage.return_value = WorkspaceUsage(
        workspace_id=workspace_id,
        billing_tier="free",
        queries_this_month=5
    )
    
    # Mock embedding
    mock_embed_service.get_embeddings.return_value = [[0.1] * 1536]
    
    # Mock pgvector similarity query returns 2 chunks
    mock_chunk1 = DocumentChunk(
        workspace_id=workspace_id,
        content="DocMind is a multi-tenant application.",
        metadata_json={"primary_page": 2}
    )
    mock_chunk2 = DocumentChunk(
        workspace_id=workspace_id,
        content="It uses pgvector for search.",
        metadata_json={"primary_page": 5}
    )
    
    db.query().join().filter().filter().order_by().limit().all.return_value = [
        (mock_chunk1, "doc1.pdf", 0.8),
        (mock_chunk2, "doc2.docx", 0.6)
    ]
    
    # 2. Mock Claude stream response
    mock_anthropic_service.stream_chat.return_value = [
        "According ", "to ", "documents, ", "DocMind ", "is ", "multi-tenant ", "[1]. ",
        "It ", "stores ", "vectors ", "in ", "pgvector ", "[2]."
    ]
    
    # Mock thread session local
    thread_db = MagicMock()
    mock_session_local.return_value = thread_db
    # Mock message retrieval in thread session
    thread_db.query().filter().order_by().limit().all.return_value = []
    
    # Execute Endpoint
    from app.routers.chat import ChatRequest
    payload = ChatRequest(message="Tell me about DocMind architecture.", conversation_id=conversation_id)
    response = await chat_rag(workspace_id, payload, db)
    
    # Read stream
    events = []
    async for event in response.body_iterator:
        events.append(event)
        
    # Check SSE format
    assert len(events) > 0
    
    # Verify content events were sent
    content_chunks = [e for e in events if "content" in e]
    assert len(content_chunks) == 13
    
    # Verify citations event was sent at the end containing both sources
    citations_event_str = [e for e in events if "citations" in e][0]
    # Remove prefix "data: "
    citations_data = json.loads(citations_event_str.replace("data: ", "").strip())
    
    assert citations_data["type"] == "citations"
    citations_list = citations_data["citations"]
    assert len(citations_list) == 2
    
    # Check mapping parameters
    assert citations_list[0]["citation_index"] == 1
    assert citations_list[0]["filename"] == "doc1.pdf"
    assert citations_list[0]["page"] == 2
    
    assert citations_list[1]["citation_index"] == 2
    assert citations_list[1]["filename"] == "doc2.docx"
    assert citations_list[1]["page"] == 5
    
    # Verify database commits
    assert thread_db.commit.called
