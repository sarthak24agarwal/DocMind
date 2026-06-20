import stripe
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from sqlalchemy.orm import Session
import uuid
import logging
from datetime import datetime, timezone

from app.config import settings
from app.database import get_db
from app.models import User, Workspace

logger = logging.getLogger(__name__)

# Initialize Stripe API Key
stripe.api_key = settings.STRIPE_API_KEY

router = APIRouter(
    prefix="/billing",
    tags=["billing"]
)

@router.post("/create-checkout-session")
def create_checkout_session(
    workspace_id: uuid.UUID,
    x_user_id: str = Header(..., description="User ID header"),
    db: Session = Depends(get_db)
):
    """
    Creates a Stripe Checkout Session for upgrading to the Pro subscription tier.
    Injects user ID and workspace ID into metadata for webhook processing.
    """
    # 1. Fetch and validate user (no limit checks needed for purchase)
    try:
        user_uuid = uuid.UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid User ID format in Header X-User-Id"
        )

    user = db.query(User).filter(User.id == user_uuid).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # 2. Check if workspace exists
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )

    # 3. Create session
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": settings.STRIPE_PRO_PRICE_ID,
                    "quantity": 1,
                }
            ],
            mode="subscription",
            customer_email=user.email,
            # In a real app these redirect to our frontend pages
            success_url="https://docmind.com/billing/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://docmind.com/billing/cancel",
            metadata={
                "workspace_id": str(workspace_id),
                "user_id": str(user.id)
            }
        )
        return {"checkout_url": checkout_session.url}
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe Checkout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate Stripe checkout flow."
        )

@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Receives and processes webhook events from Stripe (verified via signature).
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    if not sig_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe signature header."
        )

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Stripe payload."
        )
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe signature verification failed: {str(e)}"
        )

    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})

    logger.info(f"Received Stripe webhook event: {event_type}")

    if event_type == "checkout.session.completed":
        # Handle subscription creation/upgrade
        metadata = data_object.get("metadata", {})
        user_id_str = metadata.get("user_id")
        subscription_id = data_object.get("subscription")
        customer_id = data_object.get("customer")

        if not user_id_str:
            logger.error("checkout.session.completed: Missing user_id in metadata.")
            return {"status": "ignored", "reason": "missing user_id"}

        user = db.query(User).filter(User.id == uuid.UUID(user_id_str)).first()
        if user:
            user.plan = "pro"
            user.stripe_customer_id = customer_id
            user.stripe_subscription_id = subscription_id
            user.queries_limit = settings.PRO_TIER_QUERY_LIMIT
            user.payment_status = "active"
            user.payment_failed_at = None
            user.billing_cycle_anchor = datetime.now(timezone.utc)
            db.commit()
            logger.info(f"User {user_id_str} upgraded to Pro plan via Stripe.")
        else:
            logger.error(f"checkout.session.completed: User {user_id_str} not found in database.")

    elif event_type == "customer.subscription.deleted":
        # Handle cancellation/expiration
        subscription_id = data_object.get("id")
        user = db.query(User).filter(User.stripe_subscription_id == subscription_id).first()
        
        if user:
            user.plan = "free"
            user.stripe_subscription_id = None
            user.queries_limit = 100  # Revert to default free limit
            user.payment_status = "active"
            user.payment_failed_at = None
            db.commit()
            logger.info(f"User {user.id} subscription deleted. Reverted to Free plan.")

    elif event_type == "invoice.payment_failed":
        # Handle grace period activation
        subscription_id = data_object.get("subscription")
        user = db.query(User).filter(User.stripe_subscription_id == subscription_id).first()
        
        if user:
            # Shift status to past_due, stamp failure time to activate 3-day grace period
            user.payment_status = "past_due"
            user.payment_failed_at = datetime.now(timezone.utc)
            db.commit()
            logger.warning(
                f"Invoice payment failed for user {user.id}. "
                "Account marked past_due. 3-day grace period activated."
            )

    return {"status": "success"}
