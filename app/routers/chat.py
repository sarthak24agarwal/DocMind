import json
import re
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, SessionLocal
from app.models import Workspace, WorkspaceUsage, Conversation, Message, Document, DocumentChunk
from app.services.embedder import embedding_service
from app.services.anthropic_service import anthropic_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}",
    tags=["chat"]
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None


def get_or_reset_usage(db: Session, workspace_id: uuid.UUID) -> WorkspaceUsage:
    """
    Fetches the WorkspaceUsage row for a workspace, creating it on first use,
    and resetting the monthly counter whenever the calendar month has rolled over.
    """
    usage = db.query(WorkspaceUsage).filter(
        WorkspaceUsage.workspace_id == workspace_id
    ).first()

    now = datetime.now(timezone.utc)

    if usage is None:
        usage = WorkspaceUsage(
            workspace_id=workspace_id,
            billing_tier="free",
            queries_this_month=0,
            last_reset_date=now
        )
        db.add(usage)
        db.commit()
        return usage

    last_reset = usage.last_reset_date
    if last_reset.tzinfo is None:
        last_reset = last_reset.replace(tzinfo=timezone.utc)

    if last_reset.year != now.year or last_reset.month != now.month:
        usage.queries_this_month = 0
        usage.last_reset_date = now
        db.commit()

    return usage


def _usage_limit_for_tier(billing_tier: str) -> int:
    return settings.PRO_TIER_QUERY_LIMIT if billing_tier == "pro" else settings.FREE_TIER_QUERY_LIMIT


@router.post("/chat")
async def chat_rag(
    workspace_id: uuid.UUID,
    payload: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Streams a RAG-grounded response to the user's question, applying workspace-level
    monthly usage limits and a similarity threshold on retrieved chunks.
    """
    # 1. Verify workspace exists
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    # 2. Enforce workspace-level monthly query limits (auto-resets on month rollover)
    usage = get_or_reset_usage(db, workspace_id)
    limit = _usage_limit_for_tier(usage.billing_tier)
    if usage.queries_this_month >= limit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Workspace monthly query limit reached. Please upgrade your plan."
        )

    # 3. Embed user query
    try:
        query_vector = embedding_service.get_embeddings([payload.message])[0]
    except Exception as e:
        logger.error(f"Embedding query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze question. Please try again."
        )

    # 4. Query pgvector for matching chunks (k=5) scoped to workspace_id
    # similarity = 1 - cosine_distance
    cosine_distance_expr = DocumentChunk.embedding.cosine_distance(query_vector)
    similarity_expr = 1 - cosine_distance_expr

    results = db.query(
        DocumentChunk,
        Document.filename,
        similarity_expr.label("similarity")
    ).join(
        Document, Document.id == DocumentChunk.document_id
    ).filter(
        DocumentChunk.workspace_id == workspace_id
    ).filter(
        similarity_expr >= settings.SIMILARITY_THRESHOLD
    ).order_by(
        cosine_distance_expr
    ).limit(5).all()

    chunks_list = []
    for chunk, filename, similarity in results:
        chunks_list.append({
            "chunk": chunk,
            "filename": filename,
            "similarity": similarity
        })

    # 5. Handle conversation initialization
    conversation_id = payload.conversation_id
    if not conversation_id:
        conversation = Conversation(workspace_id=workspace_id, title=payload.message[:50])
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        conversation_id = conversation.id
    else:
        conversation = db.query(Conversation).filter(
            Conversation.workspace_id == workspace_id,
            Conversation.id == conversation_id
        ).first()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation history not found in this workspace."
            )

    # 6. SSE Stream Generator
    def sse_event_generator():
        # Open an independent database session for the background streaming thread
        generator_db = SessionLocal()
        full_response_text = ""
        citations_payload = []

        try:
            # Record User Message
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=payload.message
            )
            generator_db.add(user_msg)
            generator_db.commit()

            # Case A: no relevant chunks found (under threshold) -> fallback, skip calling LLM
            if not chunks_list:
                fallback_reply = (
                    "I'm sorry, I couldn't find any relevant information "
                    "in the uploaded documents to answer your question."
                )
                yield f"data: {json.dumps({'type': 'content', 'text': fallback_reply})}\n\n"
                yield f"data: {json.dumps({'type': 'citations', 'citations': []})}\n\n"

                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=fallback_reply,
                    citations=[]
                )
                generator_db.add(assistant_msg)

                thread_usage = generator_db.query(WorkspaceUsage).filter(
                    WorkspaceUsage.workspace_id == workspace_id
                ).first()
                if thread_usage:
                    thread_usage.queries_this_month += 1

                generator_db.commit()
                yield "data: [DONE]\n\n"
                return

            # Case B: relevant chunks found - construct system prompt and query Claude
            system_prompt = (
                "You are DocMind, an intelligent AI assistant grounded in the user's uploaded documents.\n"
                "Answer the user's question using ONLY the retrieved document chunks provided below.\n"
                "For every factual claim you make, you MUST cite the source chunk by appending a citation marker like [N] where N is the chunk index number (starting at 1).\n"
                "Example: 'The ingestion pipeline uses a sliding window overlap [1].'\n"
                "Do not combine citations into [1,2], use [1][2] instead.\n"
                "If the chunks do not contain enough information to answer the question, state clearly that you do not have enough information. Do not use external knowledge or hallucinate.\n\n"
                "Retrieved Document Chunks:\n"
            )

            for idx, item in enumerate(chunks_list):
                c = item["chunk"]
                f = item["filename"]
                p = c.metadata_json.get("primary_page", 1)
                system_prompt += f"\n--- Chunk [{idx + 1}] (Source: {f}, Page: {p}) ---\n{c.content}\n"

            # Load recent chat history (e.g. last 10 messages)
            recent_msgs = generator_db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.desc()).limit(11).all()
            recent_msgs.reverse()  # Keep chronologically ordered

            chat_history = []
            for msg in recent_msgs:
                if msg.id == user_msg.id:
                    continue
                chat_history.append({"role": msg.role, "content": msg.content})

            chat_history.append({"role": "user", "content": payload.message})

            # Stream Claude response
            for token in anthropic_service.stream_chat(system_prompt, chat_history):
                full_response_text += token
                yield f"data: {json.dumps({'type': 'content', 'text': token})}\n\n"

            # Extract citations after stream completes
            citation_markers = re.findall(r'\[(\d+)\]', full_response_text)
            cited_indices = sorted(list(set(int(m) for m in citation_markers)))

            for idx in cited_indices:
                if 1 <= idx <= len(chunks_list):
                    item = chunks_list[idx - 1]
                    c = item["chunk"]
                    f = item["filename"]
                    p = c.metadata_json.get("primary_page", 1)

                    citations_payload.append({
                        "citation_index": idx,
                        "filename": f,
                        "page": p,
                        "snippet": c.content[:200] + "..." if len(c.content) > 200 else c.content
                    })

            yield f"data: {json.dumps({'type': 'citations', 'citations': citations_payload})}\n\n"

            # Persist assistant response and citations
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response_text,
                citations=citations_payload
            )
            generator_db.add(assistant_msg)

            thread_usage = generator_db.query(WorkspaceUsage).filter(
                WorkspaceUsage.workspace_id == workspace_id
            ).first()
            if thread_usage:
                thread_usage.queries_this_month += 1

            generator_db.commit()
            yield "data: [DONE]\n\n"

        except Exception as e:
            generator_db.rollback()
            logger.error(f"Error in SSE stream generation: {str(e)}")
            error_payload = {"type": "error", "message": "An error occurred during response streaming."}
            yield f"data: {json.dumps(error_payload)}\n\n"
        finally:
            generator_db.close()

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream"
    )
