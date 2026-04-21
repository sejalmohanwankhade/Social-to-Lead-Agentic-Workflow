"""
main.py
CLI entrypoint for the AutoStream Social-to-Lead Conversational Agent.
Now supports OpenAI + fallback demo mode.
"""

import os
import sys

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph, AgentState


# ── ANSI colours ──────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║          AutoStream  ·  Social-to-Lead AI Agent          ║
╚══════════════════════════════════════════════════════════╝{RESET}
"""

INITIAL_GREETING = (
    "👋 Hi! I'm Aria, your AutoStream assistant.\n"
    "Ask me about features, pricing, or I can help you get started!"
)


# ── Fallback response (IMPORTANT for demo) ─────────────────────────
def fallback_response(user_input: str):
    text = user_input.lower()

    if "price" in text or "pricing" in text:
        return "Our pricing starts at $29/month. Would you like to book a demo?"

    if "feature" in text:
        return "We offer video editing automation, analytics, and integrations."

    if "demo" in text or "buy" in text:
        return "Great! Can I get your name and email to schedule a demo?"

    return "Thanks for your question! Let me help you with that."


def print_agent(text: str):
    print(f"\n{CYAN}{BOLD}Aria:{RESET} {text}\n")


def run_agent():
    # Check if OpenAI key exists
    use_llm = bool(os.getenv("OPENAI_API_KEY"))

    if not use_llm:
        print(f"{YELLOW}⚠ Running in DEMO mode (no API key){RESET}")

    # Try to build graph (if it fails, fallback mode)
    try:
        graph = build_graph()
        use_graph = True
    except Exception:
        use_graph = False

    # Initialize state
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "collecting_lead": False,
        "lead_captured": False,
        "response": "",
    }

    print(BANNER)
    print_agent(INITIAL_GREETING)

    while True:
        try:
            user_input = input(f"{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋\n")
            break

        if user_input.lower() in {"quit", "exit"}:
            break

        # Add user message
        state["messages"].append(HumanMessage(content=user_input))

        # Try real agent
        if use_llm and use_graph:
            try:
                result = graph.invoke(state)
                state.update(result)
                reply = result.get("response", "")
            except Exception:
                reply = fallback_response(user_input)
        else:
            reply = fallback_response(user_input)

        print_agent(reply)

        # Simple lead capture simulation
        if "email" in user_input.lower():
            print(f"{CYAN}Lead captured successfully! 🎉{RESET}")
            break


if __name__ == "__main__":
    run_agent()