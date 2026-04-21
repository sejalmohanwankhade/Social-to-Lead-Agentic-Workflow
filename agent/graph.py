"""
agent/graph.py
LangGraph-based agentic workflow for the AutoStream Social-to-Lead agent.

Graph nodes:
  classify_intent  →  handle_greeting
                   →  rag_response
                   →  lead_qualification  →  collect_lead_info  →  capture_lead
"""

import os
from typing import TypedDict, Annotated, Optional
import operator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from utils.intent_classifier import (
    classify_intent,
    INTENT_GREETING,
    INTENT_PRODUCT_INQUIRY,
    INTENT_HIGH_INTENT_LEAD,
)
from utils.rag_pipeline import retrieve
from tools.lead_capture import mock_lead_capture


# ─────────────────────────────────────────────
# 1.  STATE SCHEMA
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # Full conversation history (LangChain message objects)
    messages: Annotated[list, operator.add]

    # Detected intent for the current turn
    intent: str

    # Lead-collection sub-state
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Whether we are actively in lead collection mode
    collecting_lead: bool

    # Whether the lead has been captured
    lead_captured: bool

    # The agent's reply for this turn (returned to CLI loop)
    response: str


# ─────────────────────────────────────────────
# 2.  LLM INITIALISATION
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please export it before running the agent."
        )
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
        max_tokens=512,
        temperature=0.4,
    )


SYSTEM_PROMPT = """You are Aria, a friendly and knowledgeable sales assistant for AutoStream — 
an AI-powered video editing SaaS platform for content creators.

Your personality:
- Warm, enthusiastic, and concise
- Always helpful and never pushy
- Professional but approachable

Your goals:
1. Answer questions accurately using only the provided knowledge base context.
2. Detect when a user is ready to sign up and guide them through lead capture.
3. Never fabricate features, prices, or policies not in the knowledge base.

Keep responses under 120 words unless detail is genuinely required."""


# ─────────────────────────────────────────────
# 3.  NODE FUNCTIONS
# ─────────────────────────────────────────────

def node_classify_intent(state: AgentState) -> dict:
    """Classify the intent of the latest user message."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    intent = classify_intent(last_user_msg)
    return {"intent": intent}


def node_handle_greeting(state: AgentState) -> dict:
    """Respond to a casual greeting."""
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]
    reply = llm.invoke(messages).content
    return {
        "messages": [AIMessage(content=reply)],
        "response": reply,
    }


def node_rag_response(state: AgentState) -> dict:
    """Retrieve relevant KB context and generate a grounded response."""
    llm = get_llm()

    # Get the latest user message
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # RAG retrieval
    context = retrieve(last_user_msg, top_k=3)

    augmented_system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== KNOWLEDGE BASE CONTEXT ===\n{context}\n"
        f"==============================\n"
        f"Answer using ONLY the context above. If the context doesn't cover the question, "
        f"say you don't have that information."
    )

    messages = [
        SystemMessage(content=augmented_system),
        *state["messages"],
    ]
    reply = llm.invoke(messages).content
    return {
        "messages": [AIMessage(content=reply)],
        "response": reply,
    }


def node_lead_qualification(state: AgentState) -> dict:
    """Detect high intent and begin lead collection."""
    llm = get_llm()

    qualifier_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "The user has shown HIGH INTENT — they want to sign up or try AutoStream.\n"
        "Respond enthusiastically, confirm their interest, and tell them you'll get them set up.\n"
        "Then ask for their FULL NAME first. Ask for ONE thing at a time."
    )

    messages = [SystemMessage(content=qualifier_prompt), *state["messages"]]
    reply = llm.invoke(messages).content

    return {
        "messages": [AIMessage(content=reply)],
        "response": reply,
        "collecting_lead": True,
    }


def node_collect_lead_info(state: AgentState) -> dict:
    """
    Progressively collect name → email → platform from the user.
    Extracts values from the conversation history.
    """
    llm = get_llm()

    # Get all human messages for extraction
    human_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    latest_input = human_msgs[-1] if human_msgs else ""

    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")

    updates: dict = {}

    # ── Step 1: collect name ──────────────────────────────
    if not lead_name:
        # The user's latest message IS their name (we just asked for it)
        extracted_name = latest_input.strip().title()
        updates["lead_name"] = extracted_name

        ask_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Great, you've just received the user's name: '{extracted_name}'.\n"
            f"Thank them by name and now ask for their EMAIL ADDRESS."
        )
        messages = [SystemMessage(content=ask_prompt), *state["messages"]]
        reply = llm.invoke(messages).content
        updates["messages"] = [AIMessage(content=reply)]
        updates["response"] = reply
        return updates

    # ── Step 2: collect email ─────────────────────────────
    if not lead_email:
        extracted_email = latest_input.strip().lower()
        updates["lead_email"] = extracted_email

        ask_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"You have name='{lead_name}' and email='{extracted_email}'.\n"
            f"Acknowledge the email and now ask which CREATOR PLATFORM they use "
            f"(e.g., YouTube, Instagram, TikTok, LinkedIn, etc.)."
        )
        messages = [SystemMessage(content=ask_prompt), *state["messages"]]
        reply = llm.invoke(messages).content
        updates["messages"] = [AIMessage(content=reply)]
        updates["response"] = reply
        return updates

    # ── Step 3: collect platform ──────────────────────────
    if not lead_platform:
        extracted_platform = latest_input.strip().title()
        updates["lead_platform"] = extracted_platform
        updates["messages"] = [AIMessage(content="__TRIGGER_CAPTURE__")]
        updates["response"] = "__TRIGGER_CAPTURE__"
        return updates

    return {}


def node_capture_lead(state: AgentState) -> dict:
    """Call mock_lead_capture and confirm to the user."""
    name = state.get("lead_name", "")
    email = state.get("lead_email", "")
    platform = state.get("lead_platform", "")

    result = mock_lead_capture(name, email, platform)

    if result["success"]:
        reply = (
            f"🎉 You're all set, {name}! I've captured your details successfully.\n\n"
            f"**Lead ID:** `{result['lead_id']}`\n\n"
            f"Our team will reach out to {email} shortly to get your AutoStream Pro trial started. "
            f"Welcome aboard — can't wait to see what you create on {platform}! 🚀"
        )
    else:
        reply = (
            f"Hmm, something went wrong capturing your details: {result.get('error')}. "
            f"Please double-check your email and try again."
        )

    return {
        "messages": [AIMessage(content=reply)],
        "response": reply,
        "lead_captured": True,
        "collecting_lead": False,
    }


# ─────────────────────────────────────────────
# 4.  ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """Route after intent classification."""
    # If currently in lead collection mode, continue there
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return "collect_lead_info"

    intent = state.get("intent", INTENT_PRODUCT_INQUIRY)

    if intent == INTENT_GREETING:
        return "handle_greeting"
    elif intent == INTENT_HIGH_INTENT_LEAD:
        return "lead_qualification"
    else:
        return "rag_response"


def route_after_collect(state: AgentState) -> str:
    """After collecting a field, decide whether to capture or keep collecting."""
    if state.get("response") == "__TRIGGER_CAPTURE__":
        return "capture_lead"
    return END


# ─────────────────────────────────────────────
# 5.  BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", node_classify_intent)
    workflow.add_node("handle_greeting", node_handle_greeting)
    workflow.add_node("rag_response", node_rag_response)
    workflow.add_node("lead_qualification", node_lead_qualification)
    workflow.add_node("collect_lead_info", node_collect_lead_info)
    workflow.add_node("capture_lead", node_capture_lead)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Conditional routing from classify_intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "handle_greeting": "handle_greeting",
            "rag_response": "rag_response",
            "lead_qualification": "lead_qualification",
            "collect_lead_info": "collect_lead_info",
        }
    )

    # Lead qualification always moves to collect_lead_info next turn
    workflow.add_edge("lead_qualification", END)

    # Collect lead info routes to capture or ends
    workflow.add_conditional_edges(
        "collect_lead_info",
        route_after_collect,
        {
            "capture_lead": "capture_lead",
            END: END,
        }
    )

    # Terminal nodes
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("rag_response", END)
    workflow.add_edge("capture_lead", END)

    return workflow.compile()
