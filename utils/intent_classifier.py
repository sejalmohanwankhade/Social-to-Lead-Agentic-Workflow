"""
utils/intent_classifier.py
Lightweight intent classification using keyword heuristics + LLM confirmation.
Classifies user messages into one of three intents for the AutoStream agent.
"""

import re

# Intent labels
INTENT_GREETING = "greeting"
INTENT_PRODUCT_INQUIRY = "product_inquiry"
INTENT_HIGH_INTENT_LEAD = "high_intent_lead"


# --- Keyword signal sets ---

GREETING_SIGNALS = {
    "hi", "hello", "hey", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "what's up", "sup", "yo"
}

HIGH_INTENT_SIGNALS = {
    "sign up", "signup", "subscribe", "get started", "buy", "purchase",
    "want to try", "want to start", "want the pro", "want the basic",
    "i'm interested", "i am interested", "let's do it", "let me get",
    "i'll take", "i want to", "ready to", "how do i join",
    "how do i sign", "start my trial", "start a trial", "free trial",
    "can i sign", "can i subscribe", "upgrade", "enroll", "register"
}

PRODUCT_INQUIRY_SIGNALS = {
    "price", "pricing", "cost", "how much", "plan", "plans", "feature",
    "features", "include", "difference", "compare", "refund", "support",
    "resolution", "4k", "720p", "caption", "captions", "unlimited",
    "video", "videos", "basic", "pro", "policy", "trial", "cancel",
    "cancellation", "what is", "tell me about", "can you explain",
    "does it", "does autostream"
}


def classify_intent(message: str) -> str:
    """
    Classify the user's intent from their message text.

    Priority order: HIGH_INTENT > PRODUCT_INQUIRY > GREETING

    Args:
        message: Raw user message string

    Returns:
        One of: 'greeting', 'product_inquiry', 'high_intent_lead'
    """
    msg_lower = message.lower().strip()
    tokens = set(re.findall(r"\b\w+\b", msg_lower))

    # Check high intent first (highest priority)
    for signal in HIGH_INTENT_SIGNALS:
        if signal in msg_lower:
            return INTENT_HIGH_INTENT_LEAD

    # Check product inquiry
    if tokens & PRODUCT_INQUIRY_SIGNALS:
        return INTENT_PRODUCT_INQUIRY

    # Check greeting (only pure greetings)
    if tokens & GREETING_SIGNALS and len(tokens) <= 6:
        return INTENT_GREETING

    # Default to product inquiry for substantive unknown messages
    return INTENT_PRODUCT_INQUIRY
