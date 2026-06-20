import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
import logging

from app.config import settings
from app.database import get_db, engine, Base
from app.models import Workspace, Document
from app.services.r2 import r2_service
from app.tasks import process_document_ingestion
from app.routers import chat
from app.routers import billing

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(chat.router)
app.include_router(billing.router)



# Ensure tables are created (useful for dev/test environment, although migrations should handle production)
@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")

@app.post(
    "/workspaces",
    status_code=status.HTTP_201_CREATED,
    response_model=dict,
    summary="Create a new workspace (Helper for local development and testing)"
)
def create_workspace(name: str, db: Session = Depends(get_db)):
    """
    Creates a new isolated workspace.
    """
    new_workspace = Workspace(name=name)
    db.add(new_workspace)
    db.commit()
    db.refresh(new_workspace)
    return {"id": str(new_workspace.id), "name": new_workspace.name, "created_at": new_workspace.created_at}

@app.post(
    "/workspaces/{workspace_id}/documents/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=dict,
    summary="Upload and queue a document for ingestion"
)
async def upload_document(
    workspace_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Validates file size/type, uploads file to Cloudflare R2,
    inserts a 'queued' document record, and triggers the async background pipeline.
    """
    # 1. Verify workspace exists
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    # 2. Validate content type
    if file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(settings.ALLOWED_CONTENT_TYPES)}"
        )

    # 3. Validate file size by reading size from spool
    # Seek to end to get size
    await file.seek(0, 2)
    file_size = await file.tell()
    # Seek back to start so we can read it later
    await file.seek(0)

    # 0-byte file check
    if file_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty (0 bytes)."
        )

    if file_size > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.1f}MB."
        )

    # 4. Generate unique R2 key and upload file
    document_id = uuid.uuid4()
    
    # Prefixing path with workspace_id to guarantee logical division of stored items in R2
    file_key = f"workspaces/{workspace_id}/documents/{document_id}/{file.filename}"

    try:
        r2_service.upload_fileobj(file.file, file_key)
    except Exception as e:
        logger.error(f"Error uploading file to R2: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist file in storage backend."
        )

    # 5. Insert document row with status = queued
    try:
        new_doc = Document(
            id=document_id,
            workspace_id=workspace_id,
            filename=file.filename,
            file_key=file_key,
            file_size=file_size,
            content_type=file.content_type,
            status="queued"
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
    except Exception as e:
        # Cleanup R2 file if DB write fails to keep bucket clean
        logger.error(f"Failed to record document metadata in DB: {str(e)}")
        # Don't let cleanup fail the request, but log it
        try:
            r2_service.s3_client.delete_object(Bucket=settings.R2_BUCKET_NAME, Key=file_key)
        except Exception as r2_err:
            logger.warning(f"Orphan file cleanup failed in R2 for key {file_key}: {str(r2_err)}")
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record document upload details in database."
        )

    # 6. Trigger Celery Task
    process_document_ingestion.delay(str(workspace_id), str(document_id))

    return {
        "document_id": str(new_doc.id),
        "workspace_id": str(new_doc.workspace_id),
        "filename": new_doc.filename,
        "status": new_doc.status,
        "file_size": new_doc.file_size,
        "content_type": new_doc.content_type,
        "created_at": new_doc.created_at
    }

@app.get(
    "/workspaces/{workspace_id}/documents/{document_id}",
    response_model=dict,
    summary="Get status and details of a document"
)
def get_document_status(
    workspace_id: uuid.UUID,
    document_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Retrieves the ingestion status and metadata for a specific document.
    Scoping strictly filters by workspace_id first to maintain isolation.
    """
    doc = db.query(Document).filter(
        Document.workspace_id == workspace_id,
        Document.id == document_id
    ).first()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found in this workspace."
        )

    return {
        "document_id": str(doc.id),
        "workspace_id": str(doc.workspace_id),
        "filename": doc.filename,
        "status": doc.status,
        "error_message": doc.error_message,
        "file_size": doc.file_size,
        "content_type": doc.content_type,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at
    }
