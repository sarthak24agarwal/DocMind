import pytest
from unittest.mock import MagicMock, patch
import uuid
import stripe
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException

from app.dependencies import verify_query_limits
from app.routers.billing import create_checkout_session, stripe_webhook
from app.tasks import reset_monthly_query_counters
from app.models import User, Workspace

# ----------------------------------------------------
# 1. Checkout Session Tests
# ----------------------------------------------------

@patch("app.routers.billing.stripe.checkout.Session.create")
def test_create_checkout_session_success(mock_stripe_create):
    db = MagicMock()
    workspace_id = uuid.uuid4()
    user_id = uuid.uuid4()
    
    # Mock user and workspace in database
    mock_user = User(id=user_id, email="test@docmind.com")
    mock_workspace = Workspace(id=workspace_id)
    
    # Mock database first() returns
    db.query().filter().first.side_effect = [mock_user, mock_workspace]
    
    # Mock stripe response URL
    mock_session = MagicMock()
    mock_session.url = "https://checkout.stripe.com/pay/cs_test_123"
    mock_stripe_create.return_value = mock_session
    
    response = create_checkout_session(workspace_id, str(user_id), db)
    
    assert response["checkout_url"] == "https://checkout.stripe.com/pay/cs_test_123"
    mock_stripe_create.assert_called_once()
    # Confirm metadata holds our workspace and user link
    metadata = mock_stripe_create.call_args[1]["metadata"]
    assert metadata["workspace_id"] == str(workspace_id)
    assert metadata["user_id"] == str(user_id)

# ----------------------------------------------------
# 2. Webhook Event Processing Tests
# ----------------------------------------------------

@patch("app.routers.billing.stripe.Webhook.construct_event")
@pytest.mark.asyncio
async def test_stripe_webhook_upgrade_pro(mock_construct):
    db = MagicMock()
    user_id = uuid.uuid4()
    
    # Mock constructed event: checkout.session.completed
    mock_event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "customer": "cus_123",
                "subscription": "sub_123",
                "metadata": {
                    "user_id": str(user_id),
                    "workspace_id": str(uuid.uuid4())
                }
            }
        }
    }
    mock_construct.return_value = mock_event
    
    # Mock database lookup
    mock_user = User(id=user_id, plan="free", queries_limit=100)
    db.query().filter().first.return_value = mock_user
    
    # Setup mock request
    mock_request = MagicMock()
    mock_request.body = pytest.AsyncMock(return_value=b"raw_body")
    mock_request.headers = {"stripe-signature": "valid_sig"}
    
    response = await stripe_webhook(mock_request, db)
    
    assert response["status"] == "success"
    # User upgraded to Pro details
    assert mock_user.plan == "pro"
    assert mock_user.stripe_customer_id == "cus_123"
    assert mock_user.stripe_subscription_id == "sub_123"
    assert mock_user.queries_limit == 5000
    assert mock_user.payment_status == "active"
    assert db.commit.called

@patch("app.routers.billing.stripe.Webhook.construct_event")
@pytest.mark.asyncio
async def test_stripe_webhook_downgrade_free(mock_construct):
    db = MagicMock()
    
    mock_event = {
        "type": "customer.subscription.deleted",
        "data": {
            "object": {
                "id": "sub_123"
            }
        }
    }
    mock_construct.return_value = mock_event
    
    mock_user = User(id=uuid.uuid4(), plan="pro", stripe_subscription_id="sub_123", queries_limit=5000)
    db.query().filter().first.return_value = mock_user
    
    mock_request = MagicMock()
    mock_request.body = pytest.AsyncMock(return_value=b"raw_body")
    mock_request.headers = {"stripe-signature": "valid_sig"}
    
    response = await stripe_webhook(mock_request, db)
    
    assert response["status"] == "success"
    # Reverted to free tier
    assert mock_user.plan == "free"
    assert mock_user.stripe_subscription_id is None
    assert mock_user.queries_limit == 100
    assert db.commit.called

@patch("app.routers.billing.stripe.Webhook.construct_event")
@pytest.mark.asyncio
async def test_stripe_webhook_payment_failed_grace_period(mock_construct):
    db = MagicMock()
    
    mock_event = {
        "type": "invoice.payment_failed",
        "data": {
            "object": {
                "subscription": "sub_123"
            }
        }
    }
    mock_construct.return_value = mock_event
    
    mock_user = User(id=uuid.uuid4(), plan="pro", stripe_subscription_id="sub_123", payment_status="active")
    db.query().filter().first.return_value = mock_user
    
    mock_request = MagicMock()
    mock_request.body = pytest.AsyncMock(return_value=b"raw_body")
    mock_request.headers = {"stripe-signature": "valid_sig"}
    
    response = await stripe_webhook(mock_request, db)
    
    assert response["status"] == "success"
    # Status becomes past_due, failures timestamped to invoke grace period
    assert mock_user.payment_status == "past_due"
    assert mock_user.payment_failed_at is not None
    assert db.commit.called

# ----------------------------------------------------
# 3. Limit Enforcement Dependency Tests
# ----------------------------------------------------

def test_verify_query_limits_success():
    db = MagicMock()
    user_id = uuid.uuid4()
    
    # User is within limits
    mock_user = User(id=user_id, queries_used_this_month=50, queries_limit=100, payment_status="active")
    db.query().filter().first.return_value = mock_user
    
    result = verify_query_limits(str(user_id), db)
    assert result == mock_user

def test_verify_query_limits_exceeded():
    db = MagicMock()
    user_id = uuid.uuid4()
    
    # User has reached limit
    mock_user = User(id=user_id, queries_used_this_month=100, queries_limit=100, payment_status="active")
    db.query().filter().first.return_value = mock_user
    
    with pytest.raises(HTTPException) as exc_info:
        verify_query_limits(str(user_id), db)
    assert exc_info.value.status_code == 402
    assert "limit reached" in exc_info.value.detail

def test_verify_query_limits_past_due_in_grace_period():
    db = MagicMock()
    user_id = uuid.uuid4()
    
    # Payment failed 1 day ago (grace period is 3 days)
    failed_at = datetime.now(timezone.utc) - timedelta(days=1)
    mock_user = User(
        id=user_id,
        queries_used_this_month=10,
        queries_limit=100,
        payment_status="past_due",
        payment_failed_at=failed_at
    )
    db.query().filter().first.return_value = mock_user
    
    # Should allow request since it is within grace period
    result = verify_query_limits(str(user_id), db)
    assert result == mock_user

def test_verify_query_limits_past_due_grace_expired():
    db = MagicMock()
    user_id = uuid.uuid4()
    
    # Payment failed 4 days ago (grace period has expired)
    failed_at = datetime.now(timezone.utc) - timedelta(days=4)
    mock_user = User(
        id=user_id,
        queries_used_this_month=10,
        queries_limit=100,
        payment_status="past_due",
        payment_failed_at=failed_at
    )
    db.query().filter().first.return_value = mock_user
    
    # Should block request since grace period has expired
    with pytest.raises(HTTPException) as exc_info:
        verify_query_limits(str(user_id), db)
    assert exc_info.value.status_code == 402
    assert "grace period of 3 days has expired" in exc_info.value.detail

# ----------------------------------------------------
# 4. Scheduled Reset Task Tests
# ----------------------------------------------------

@patch("app.tasks.SessionLocal")
def test_reset_monthly_query_counters_beat_job(mock_session_local):
    db = MagicMock()
    mock_session_local.return_value = db
    
    # Setup users
    user_free = User(plan="free", queries_used_this_month=50, queries_limit=100)
    
    # Pro user whose anchor day matches today (20th of the month)
    anchor_today = datetime.now(timezone.utc).replace(day=20)
    user_pro_active = User(
        plan="pro",
        queries_used_this_month=2000,
        queries_limit=5000,
        billing_cycle_anchor=anchor_today
    )
    
    # Pro user whose anchor day is different (e.g. 5th of the month)
    anchor_other = datetime.now(timezone.utc).replace(day=5)
    user_pro_inactive = User(
        plan="pro",
        queries_used_this_month=150,
        queries_limit=5000,
        billing_cycle_anchor=anchor_other
    )
    
    # Mock query lookups
    db.query().filter().all.side_effect = [
        [user_free],  # 1st of month query (returns empty if not day 1)
        [user_pro_active, user_pro_inactive]  # Pro query
    ]
    
    # Force mock "now" inside tasks to be day 20 (not day 1, so free users shouldn't reset)
    with patch("app.tasks.datetime") as mock_datetime:
        mock_now = MagicMock()
        mock_now.day = 20
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        reset_monthly_query_counters()
        
        # Free user should NOT be reset (since today is day 20, not day 1)
        assert user_free.queries_used_this_month == 50
        
        # Pro user matching day 20 should be reset to 0
        assert user_pro_active.queries_used_this_month == 0
        
        # Pro user matching day 5 should NOT be reset
        assert user_pro_inactive.queries_used_this_month == 150
        
        assert db.commit.called
