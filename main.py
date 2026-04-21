"""
main.py
CLI entrypoint for the AutoStream Social-to-Lead Conversational Agent.
Powered by LangGraph + Claude Haiku + local RAG.

Usage:
    python main.py
"""

import os
import sys

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph, AgentState


# ── ANSI colours for the CLI ──────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║          AutoStream  ·  Social-to-Lead AI Agent          ║
║                  Powered by Inflx  ·  ServiceHive        ║
╚══════════════════════════════════════════════════════════╝{RESET}

Type {GREEN}your message{RESET} and press Enter to chat.
Type {YELLOW}'quit'{RESET} or {YELLOW}'exit'{RESET} to end the session.
"""

INITIAL_GREETING = (
    "👋 Hi there! I'm **Aria**, your AutoStream assistant.\n"
    "I can help you learn about our video editing plans, pricing, and policies — "
    "or get you signed up today! How can I help you? 🎬"
)


def print_agent(text: str):
    # Strip internal sentinel
    if text == "__TRIGGER_CAPTURE__":
        return
    print(f"\n{CYAN}{BOLD}Aria:{RESET} {text}\n")


def print_user(text: str):
    print(f"{GREEN}You:{RESET} {text}")


def run_agent():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            f"\n{YELLOW}⚠  ANTHROPIC_API_KEY is not set.\n"
            f"   Export it with:  export ANTHROPIC_API_KEY=sk-ant-...\n{RESET}"
        )
        sys.exit(1)

    graph = build_graph()

    # Initialise state
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
            print(f"\n{YELLOW}Session ended. Goodbye! 👋{RESET}\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print(f"\n{YELLOW}Thanks for chatting! See you on AutoStream. 🎬{RESET}\n")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
        try:
            result = graph.invoke(state)
        except Exception as e:
            print(f"\n{YELLOW}⚠  Agent error: {e}{RESET}\n")
            continue

        # Update persistent state
        state.update(result)

        # Print agent response
        agent_reply = result.get("response", "")
        print_agent(agent_reply)

        # End session after successful lead capture
        if result.get("lead_captured"):
            print(f"{CYAN}{'─' * 58}{RESET}")
            print(f"  Session complete — lead successfully captured!")
            print(f"{CYAN}{'─' * 58}{RESET}\n")
            break


if __name__ == "__main__":
    run_agent()
