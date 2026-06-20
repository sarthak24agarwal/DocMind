import uuid
from sqlalchemy import Column, String, Integer, BigInteger, ForeignKey, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database import Base
from app.config import settings

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    plan = Column(String(50), default="free", nullable=False)  # free, pro
    stripe_customer_id = Column(String(255), nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    queries_used_this_month = Column(Integer, default=0, nullable=False)
    queries_limit = Column(Integer, default=100, nullable=False)
    payment_status = Column(String(50), default="active", nullable=False)  # active, past_due
    payment_failed_at = Column(DateTime(timezone=True), nullable=True)
    billing_cycle_anchor = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_workspaces_id", "id"),
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    file_key = Column(String(1024), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(255), nullable=False)
    status = Column(String(50), default="queued", nullable=False)  # queued, processing, ready, failed
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_documents_workspace_id_id", "workspace_id", "id"),
    )

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    
    # Store pgvector embedding with the configured dimension (e.g. 1536)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)
    
    # Store metadata such as page number, character offset, token count, etc.
    metadata_json = Column("metadata", JSONB, nullable=False, server_default="{}")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        # Ensure fast queries when searching documents inside a specific workspace
        Index("ix_document_chunks_workspace_document", "workspace_id", "document_id"),
        # Compound index for uniqueness/idempotency checks
        Index("ix_document_chunks_workspace_doc_index", "workspace_id", "document_id", "chunk_index"),
    )

class WorkspaceUsage(Base):
    __tablename__ = "workspace_usages"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    billing_tier = Column(String(50), default="free", nullable=False)
    queries_this_month = Column(Integer, default=0, nullable=False)
    last_reset_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_workspace_usages_workspace_id", "workspace_id"),
    )

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), default="New Conversation", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_conversations_workspace_id_id", "workspace_id", "id"),
    )

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # user, assistant
    content = Column(String, nullable=False)
    
    # Store parsed citation mappings back to source chunks (e.g. filename, page, index, snippet)
    citations = Column(JSONB, nullable=True, server_default="[]")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_messages_conversation_id_created", "conversation_id", "created_at"),
    )

