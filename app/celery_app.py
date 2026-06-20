from celery import Celery
from celery.schedules import crontab
from app.config import settings

celery_app = Celery(
    "docmind_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"]
)

# Configure Celery performance and security parameters
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Clean up task results after 1 hour
    result_expires=3600,
    # Prevent task prefetching to distribute heavy document tasks evenly across workers
    worker_prefetch_multiplier=1,
)

# Register Celery Beat schedule for resetting user query usage counters daily at midnight (00:00) UTC
celery_app.conf.beat_schedule = {
    "reset-monthly-query-counters-daily": {
        "task": "app.tasks.reset_monthly_query_counters",
        "schedule": crontab(hour=0, minute=0),
    }
}


if __name__ == "__main__":
    celery_app.start()
