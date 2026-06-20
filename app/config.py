from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "DocMind Backend"
    DEBUG: bool = False

    # Database Settings
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/docmind",
        description="PostgreSQL connection string with pgvector support"
    )

    # Redis / Celery Settings
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL used for Celery broker and backend"
    )

    # Cloudflare R2 / S3 Settings
    R2_ENDPOINT_URL: str = Field(
        ...,
        description="Cloudflare R2 Endpoint URL (or other S3 compatible storage endpoint)"
    )
    R2_ACCESS_KEY_ID: str = Field(..., description="R2 / AWS Access Key ID")
    R2_SECRET_ACCESS_KEY: str = Field(..., description="R2 / AWS Secret Access Key")
    R2_BUCKET_NAME: str = Field(..., description="S3/R2 Bucket name for document storage")

    # OpenAI Settings (for Embeddings and/or Claude API if needed, let's support standard OpenAI embedding)
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key used for embeddings generation")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_BATCH_SIZE: int = 100

    # Anthropic Claude Settings
    ANTHROPIC_API_KEY: str = Field(..., description="Anthropic API Key for Claude chat model")
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20240620"

    # RAG Settings
    SIMILARITY_THRESHOLD: float = Field(0.35, description="Cosine similarity threshold for retrieved chunks")
    FREE_TIER_QUERY_LIMIT: int = Field(100, description="Monthly chat query limit for free workspaces")

    # Stripe Settings
    STRIPE_API_KEY: str = Field(..., description="Stripe Secret API Key")
    STRIPE_WEBHOOK_SECRET: str = Field(..., description="Stripe Webhook Signing Secret")
    STRIPE_PRO_PRICE_ID: str = Field(..., description="Stripe Price ID for Pro plan")
    PRO_TIER_QUERY_LIMIT: int = Field(5000, description="Monthly query limit for Pro users")

    # Ingestion Configuration

    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB default limit

    ALLOWED_CONTENT_TYPES: list[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
        "text/plain"
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings singleton
settings = Settings()
