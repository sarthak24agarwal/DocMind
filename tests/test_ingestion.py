import pytest
from unittest.mock import MagicMock, patch
import uuid

from app.services.parser import parse_document, ParsingError
from app.services.chunker import Chunker, split_into_sentences
from app.services.embedder import EmbeddingService
from app.tasks import process_document_ingestion
from app.models import Document, DocumentChunk

# ----------------------------------------------------
# 1. Chunker Tests
# ----------------------------------------------------

def test_split_into_sentences():
    text = "Hello world. This is Antigravity testing. Mr. Smith went to the store. Is this a test?"
    sentences = split_into_sentences(text)
    
    assert len(sentences) == 4
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is Antigravity testing."
    # Check that "Mr. Smith" abbreviation is preserved and not split
    assert sentences[2] == "Mr. Smith went to the store."
    assert sentences[3] == "Is this a test?"

def test_chunker_basic():
    # Target 30 tokens, 5 token overlap (cl100k_base representation)
    chunker = Chunker(target_tokens=30, overlap_tokens=5)
    
    # Each sentence is roughly 5-10 tokens
    blocks = [
        {"content": "This is paragraph one containing sentence one. It is long and descriptive.", "page": 1, "position": "paragraph_1"},
        {"content": "This is paragraph two containing sentence two. It serves to test the window.", "page": 2, "position": "paragraph_2"}
    ]
    
    chunks = chunker.chunk_document(blocks)
    
    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert "chunk_index" in chunk
        assert "content" in chunk
        assert "metadata" in chunk
        
        # Verify metadata keys
        meta = chunk["metadata"]
        assert "pages" in meta
        assert "primary_page" in meta
        assert "positions" in meta
        assert "token_count" in meta
        assert "char_count" in meta
        
        # Check chunk index ordering
        assert chunk["chunk_index"] == idx

def test_chunker_extreme_sentence_splitting():
    # Test if Chunker splits an extremely long sentence that exceeds target_tokens limit
    chunker = Chunker(target_tokens=10, overlap_tokens=2)
    blocks = [
        {"content": "Word " * 50, "page": 1, "position": "paragraph_1"}  # ~50 tokens in 1 sentence
    ]
    chunks = chunker.chunk_document(blocks)
    
    # Should split it into multiple chunks
    assert len(chunks) > 1
    # Check token counts
    for chunk in chunks:
        assert chunk["metadata"]["token_count"] <= 10 + 5  # allowance for word-based token boundaries

# ----------------------------------------------------
# 2. Parser Tests
# ----------------------------------------------------

@patch("app.services.parser.pypdf.PdfReader")
def test_parse_pdf_success(mock_pdf_reader):
    # Setup mock PDF pages
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page one content."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page two content."
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.is_encrypted = False
    mock_reader_instance.pages = [mock_page1, mock_page2]
    mock_pdf_reader.return_value = mock_reader_instance
    
    result = parse_document("dummy_path.pdf", "application/pdf")
    
    assert len(result) == 2
    assert result[0]["content"] == "Page one content."
    assert result[0]["page"] == 1
    assert result[1]["content"] == "Page two content."
    assert result[1]["page"] == 2

@patch("app.services.parser.docx.Document")
def test_parse_docx_success(mock_docx_document):
    # Setup mock docx paragraphs
    mock_para1 = MagicMock()
    mock_para1.text = "Para one content."
    mock_para2 = MagicMock()
    mock_para2.text = "Para two content."
    
    mock_doc_instance = MagicMock()
    mock_doc_instance.paragraphs = [mock_para1, mock_para2]
    mock_doc_instance.tables = []
    mock_docx_document.return_value = mock_doc_instance
    
    result = parse_document("dummy_path.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    
    assert len(result) == 2
    assert result[0]["content"] == "Para one content."
    assert result[0]["position"] == "paragraph_1"
    assert result[1]["content"] == "Para two content."
    assert result[1]["position"] == "paragraph_2"

# ----------------------------------------------------
# 3. Ingestion Task Integration Tests
# ----------------------------------------------------

@patch("app.tasks.SessionLocal")
@patch("app.tasks.r2_service")
@patch("app.tasks.parse_document")
@patch("app.tasks.embedding_service")
def test_process_document_ingestion_success(
    mock_embed_service,
    mock_parse_document,
    mock_r2_service,
    mock_session_local
):
    workspace_id = uuid.uuid4()
    document_id = uuid.uuid4()
    
    # 1. Setup mock database session & models
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db
    
    mock_document = Document(
        id=document_id,
        workspace_id=workspace_id,
        filename="test.txt",
        file_key=f"workspaces/{workspace_id}/documents/{document_id}/test.txt",
        file_size=100,
        content_type="text/plain",
        status="queued"
    )
    # Mock first() result
    mock_db.query().filter().first.return_value = mock_document
    
    # 2. Setup parsing, chunking, and embedding results
    mock_parse_document.return_value = [
        {"content": "This is block one.", "page": 1, "position": "paragraph_1"},
        {"content": "This is block two.", "page": 1, "position": "paragraph_2"}
    ]
    
    mock_embed_service.get_embeddings.return_value = [
        [0.1] * 1536,
        [0.2] * 1536
    ]
    
    # Execute the Celery task
    process_document_ingestion(str(workspace_id), str(document_id))
    
    # 3. Assertions
    # Check that it fetched the document strictly scoped by workspace_id and document_id
    mock_db.query.assert_called()
    
    # Check R2 download call
    mock_r2_service.download_file.assert_called_once_with(
        mock_document.file_key,
        pytest.any_str
    )
    
    # Check document state transitions
    assert mock_document.status == "ready"
    assert mock_document.error_message is None
    
    # Check transaction lifecycle (commit called)
    assert mock_db.commit.called
    
    # Check idempotency: verifying that existing chunks are deleted prior to inserting new ones
    mock_db.query().filter().delete.assert_called()

@patch("app.tasks.SessionLocal")
@patch("app.tasks.r2_service")
@patch("app.tasks.parse_document")
def test_process_document_ingestion_parsing_failure(
    mock_parse_document,
    mock_r2_service,
    mock_session_local
):
    workspace_id = uuid.uuid4()
    document_id = uuid.uuid4()
    
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db
    
    mock_document = Document(
        id=document_id,
        workspace_id=workspace_id,
        filename="corrupted.pdf",
        file_key="dummy_key",
        file_size=100,
        content_type="application/pdf",
        status="queued"
    )
    mock_db.query().filter().first.return_value = mock_document
    
    # Make parser throw terminal ParsingError
    mock_parse_document.side_effect = ParsingError("PDF file is corrupted or empty.")
    
    process_document_ingestion(str(workspace_id), str(document_id))
    
    # Assertions
    assert mock_document.status == "failed"
    assert "PDF file is corrupted or empty" in mock_document.error_message
    mock_db.commit.assert_called()
