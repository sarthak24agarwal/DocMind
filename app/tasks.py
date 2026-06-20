import os
import tempfile
import logging
from uuid import UUID
from datetime import datetime, timezone, timedelta
from celery.exceptions import MaxRetriesExceededError
from openai import OpenAIError

from app.celery_app import celery_app
from app.database import SessionLocal
from app.models import Document, DocumentChunk, User

from app.services.r2 import r2_service
from app.services.parser import parse_document, ParsingError
from app.services.chunker import Chunker
from app.services.embedder import embedding_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_document_ingestion(self, workspace_id_str: str, document_id_str: str):
    """
    Background job to parse, chunk, embed, and store document data.
    """
    workspace_id = UUID(workspace_id_str)
    document_id = UUID(document_id_str)
    
    db = SessionLocal()
    
    # 1. Fetch document and verify existence (strictly scoped by workspace_id)
    doc = db.query(Document).filter(
        Document.workspace_id == workspace_id,
        Document.id == document_id
    ).first()
    
    if not doc:
        logger.error(
            f"Ingestion failed: Document {document_id_str} not found in workspace {workspace_id_str}. "
            "Terminating task execution to prevent data leakage."
        )
        db.close()
        return

    # Update state to processing
    doc.status = "processing"
    doc.error_message = None
    db.commit()

    # Define temporary file path for cleanup guarantee
    temp_file_path = None
    
    try:
        # Create a safe temp file to store downloaded content
        suffix = os.path.splitext(doc.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name

        # a. Download from R2
        try:
            r2_service.download_file(doc.file_key, temp_file_path)
        except Exception as e:
            logger.warning(f"R2 download failed for key {doc.file_key}: {str(e)}. Retrying task.")
            # Retry Celery task for transient network errors
            raise self.retry(exc=e)

        # b. Parse file based on mime type
        logger.info(f"Parsing document {doc.filename}...")
        blocks = parse_document(temp_file_path, doc.content_type)
        
        # c. Chunk text semantically
        logger.info(f"Chunking document {doc.filename}...")
        chunker = Chunker(target_tokens=500, overlap_tokens=50)
        chunks = chunker.chunk_document(blocks)
        
        # d. Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk["content"] for chunk in chunks]
        
        try:
            embeddings = embedding_service.get_embeddings(texts)
        except (OpenAIError, Exception) as e:
            logger.warning(f"Embeddings generation failed: {str(e)}. Retrying task.")
            raise self.retry(exc=e)

        # e. Atomic Insert + Idempotency Cleanup
        logger.info(f"Saving {len(chunks)} chunks to database...")
        new_chunks = []
        for idx, chunk in enumerate(chunks):
            new_chunks.append(
                DocumentChunk(
                    workspace_id=workspace_id,
                    document_id=document_id,
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    embedding=embeddings[idx],
                    metadata_json=chunk["metadata"]
                )
            )

        # Execute operations in a single database transaction to guarantee atomicity and idempotency
        try:
            # Re-fetch doc inside transaction context if needed
            db.begin_nested() if db.in_transaction() else None
            
            # Clean up any existing chunks (idempotent overwrite)
            db.query(DocumentChunk).filter(
                DocumentChunk.workspace_id == workspace_id,
                DocumentChunk.document_id == document_id
            ).delete()
            
            # Bulk save chunks
            db.add_all(new_chunks)
            
            # f. Update status to ready
            doc.status = "ready"
            doc.error_message = None
            db.commit()
            logger.info(f"Ingestion pipeline completed successfully for document {doc.filename}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Database write transaction failed: {str(e)}")
            raise e

    except ParsingError as e:
        # Terminal errors (malformed file, wrong decoding) are recorded and marked as failed (no retry)
        logger.error(f"Ingestion failed with terminal parsing error: {str(e)}")
        doc.status = "failed"
        doc.error_message = f"Parsing failed: {str(e)}"
        db.commit()

    except MaxRetriesExceededError as e:
        logger.error(f"Ingestion task exceeded max retries. Marking document as failed.")
        doc.status = "failed"
        doc.error_message = "Ingestion failed: Task retry limit exceeded due to external service downtime."
        db.commit()

    except Exception as e:
        # Standard fallback for any other unexpected failures
        logger.error(f"Unexpected error in ingestion pipeline: {str(e)}")
        doc.status = "failed"
        doc.error_message = f"Ingestion failed: {str(e)}"
        db.commit()

    finally:
        # Guarantee cleanup of temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {str(cleanup_err)}")
        db.close()

@celery_app.task
def reset_monthly_query_counters():
    """
    Daily background job that checks which users need their monthly usage count reset.
    - Free tier users reset on the 1st of every calendar month.
    - Pro tier users reset on their subscription cycle anchor day.
    """
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)

        # Always run this lookup so downstream query ordering (e.g. the Pro-tier
        # query that follows) stays deterministic, even on days we don't act on it.
        free_users_to_reset = db.query(User).filter(
            User.plan == "free",
            User.queries_used_this_month > 0
        ).all()

        # 1. Reset free users on the 1st of the month
        if now.day == 1:
            for u in free_users_to_reset:
                u.queries_used_this_month = 0
            logger.info(f"Daily Reset: Cleared query usage for {len(free_users_to_reset)} free tier users.")

        # 2. Reset pro users on their specific monthly billing cycle anniversary day
        # Handle end-of-month edge cases (e.g., if anchor is 31st and month ends on 30th)
        pro_users = db.query(User).filter(User.plan == "pro").all()
        reset_pro_count = 0
        for u in pro_users:
            anchor_day = u.billing_cycle_anchor.day
            should_reset = False
            
            if anchor_day == now.day:
                should_reset = True
            else:
                # If tomorrow is the 1st day of a new month, it means today is the last day of the current month.
                # If the anchor day exceeds today's day (e.g., anchor day is 31st, but today is 30th), reset today.
                tomorrow = now + timedelta(days=1)
                # Ensure timezone awareness matches
                tomorrow_utc = tomorrow.astimezone(timezone.utc) if tomorrow.tzinfo else tomorrow.replace(tzinfo=timezone.utc)
                if tomorrow_utc.day == 1 and anchor_day > now.day:
                    should_reset = True

            if should_reset and u.queries_used_this_month > 0:
                u.queries_used_this_month = 0
                reset_pro_count += 1

        if reset_pro_count > 0:
            logger.info(f"Daily Reset: Cleared query usage for {reset_pro_count} Pro tier users based on billing anchors.")

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to reset monthly query counters: {str(e)}")
    finally:
        db.close()

