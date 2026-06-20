from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.orm import Session
import uuid
from datetime import datetime, timezone, timedelta

from app.database import get_db
from app.models import User

def verify_query_limits(
    x_user_id: str = Header(..., description="Authentication header mapping to User ID"),
    db: Session = Depends(get_db)
):
    """
    Enforces user query usage constraints and payment status.
    Blocks requests if user is over their limit or past their 3-day grace period.
    """
    try:
        user_uuid = uuid.UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header 'X-User-Id' must be a valid UUID format."
        )

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorized User profile not found."
        )

    # 1. Enforce query limit boundaries
    if user.queries_used_this_month >= user.queries_limit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Monthly query limit reached. Please upgrade your plan."
        )

    # 2. Enforce payment grace periods (past_due check)
    if user.payment_status == "past_due":
        if user.payment_failed_at:
            grace_expiry = user.payment_failed_at + timedelta(days=3)
            # Compare in UTC timezone
            now_utc = datetime.now(timezone.utc)
            # Ensure payment_failed_at is timezone-aware or make compared times timezone-naive
            failed_at_aware = user.payment_failed_at
            if failed_at_aware.tzinfo is None:
                failed_at_aware = failed_at_aware.replace(tzinfo=timezone.utc)
                grace_expiry = failed_at_aware + timedelta(days=3)
                
            if now_utc > grace_expiry:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Subscription payment past due. grace period of 3 days has expired. Please update your card details."
                )

    return user
