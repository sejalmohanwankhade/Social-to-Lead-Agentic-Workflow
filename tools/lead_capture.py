"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
Simulates sending lead data to a CRM or backend API.
"""

import json
import datetime
from typing import Optional


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.
    
    In production, this would POST to a CRM endpoint (e.g., HubSpot, Salesforce)
    or an internal ServiceHive/Inflx backend.

    Args:
        name     : Full name of the lead
        email    : Email address of the lead
        platform : Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with status and lead ID
    """
    # Validate inputs
    if not name or not email or not platform:
        return {
            "success": False,
            "error": "Missing required fields: name, email, or platform"
        }

    if "@" not in email or "." not in email.split("@")[-1]:
        return {
            "success": False,
            "error": f"Invalid email format: {email}"
        }

    # Simulate a unique lead ID
    lead_id = f"LEAD-{abs(hash(email)) % 100000:05d}"
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    lead_record = {
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "product_interest": "AutoStream Pro Plan",
        "source": "Inflx AI Agent",
        "captured_at": timestamp
    }

    # Print to simulate backend confirmation (would be an API call in production)
    print("\n" + "=" * 55)
    print("  ✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Lead captured successfully: {name}, {email}, {platform}")
    print("-" * 55)
    print(f"  Lead ID   : {lead_id}")
    print(f"  Timestamp : {timestamp}")
    print("=" * 55 + "\n")

    return {
        "success": True,
        "lead_id": lead_id,
        "message": f"Lead for {name} captured successfully.",
        "data": lead_record
    }
