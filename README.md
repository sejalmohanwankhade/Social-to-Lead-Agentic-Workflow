# AutoStream · Social-to-Lead AI Agent
### Machine Learning Intern Assignment — ServiceHive / Inflx

> A production-grade conversational AI agent that converts social media intent into qualified business leads using LangGraph, Claude Haiku, and local RAG.

---

## 📁 Project Structure

```
autostream-agent/
├── main.py                          # CLI entrypoint
├── requirements.txt
├── README.md
├── knowledge_base/
│   └── autostream_kb.json           # Local knowledge base (pricing, features, policies)
├── agent/
│   └── graph.py                     # LangGraph state machine (nodes + edges)
├── tools/
│   └── lead_capture.py              # mock_lead_capture() tool
└── utils/
    ├── rag_pipeline.py              # Local RAG retrieval
    └── intent_classifier.py         # Keyword-based intent classifier
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- An [Anthropic API key](https://console.anthropic.com/) (Claude Haiku)

### 1. Clone and install

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
# Windows PowerShell:
# $env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Or create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run the agent

```bash
python main.py
```

### Example session

```
You: Hi there!
Aria: Hey! I'm Aria, your AutoStream assistant. How can I help?

You: What's the difference between the Basic and Pro plans?
Aria: Great question! Basic is $29/month — 10 videos, 720p. Pro is $79/month —
      unlimited videos, 4K, AI captions, and 24/7 support.

You: That sounds great, I want to sign up for the Pro plan for my YouTube channel.
Aria: Amazing! Let's get you set up. What's your full name?

You: Alex Johnson
Aria: Thanks Alex! What's your email address?

You: alex@example.com
Aria: Perfect! And which creator platform do you mainly use?

You: YouTube
Aria: 🎉 You're all set, Alex! Lead captured. Our team will reach out shortly!

[Lead captured: Alex Johnson | alex@example.com | YouTube]
```

---

## 🏗 Architecture Explanation (~220 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because this workflow requires **deterministic, inspectable state transitions** rather than autonomous multi-agent negotiation. The lead capture flow has a strict sequential structure (greet → retrieve → qualify → collect name → collect email → collect platform → capture), and LangGraph's explicit `StateGraph` makes this flow transparent and testable. Each node is a pure function; the router decides the next step based on typed state — making debugging and auditing straightforward.

### How State Is Managed

`AgentState` is a `TypedDict` that persists across all turns in a single session. It tracks:
- **`messages`** — full `HumanMessage`/`AIMessage` history (LangChain objects), giving the LLM complete conversational memory across 5–6+ turns
- **`intent`** — classified intent per turn
- **`lead_name / lead_email / lead_platform`** — progressively filled during lead collection
- **`collecting_lead / lead_captured`** — boolean flags controlling graph routing

The graph routes through `classify_intent` on every turn. If `collecting_lead=True`, the router bypasses intent classification and goes directly to `collect_lead_info`, ensuring the three fields are gathered sequentially without re-triggering intent logic.

The LLM (Claude Haiku) receives the full `messages` list in every call, giving it native conversational context without an external memory store.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business API (Cloud API)** via Meta and expose your agent over HTTPS webhooks.

### Architecture

```
WhatsApp User
     │
     ▼
Meta WhatsApp Cloud API
     │  (POST webhook)
     ▼
Your Server  ──►  verify_token check  ──►  extract message text
     │
     ▼
AutoStream LangGraph Agent  ──►  generate response
     │
     ▼
WhatsApp Cloud API  ──►  POST /messages  ──►  User receives reply
```

### Implementation Steps

**1. Register a Meta Business App**
- Go to [developers.facebook.com](https://developers.facebook.com) and create a Business App
- Add the WhatsApp product and obtain a `Phone Number ID` and `Access Token`

**2. Set up a webhook server (FastAPI example)**

```python
from fastapi import FastAPI, Request
from agent.graph import build_graph, AgentState
from langchain_core.messages import HumanMessage

app = FastAPI()
graph = build_graph()

# In-memory session store (use Redis in production)
sessions: dict[str, AgentState] = {}

VERIFY_TOKEN = "your-webhook-verify-token"

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = dict(request.query_params)
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(params["hub.challenge"])
    return {"error": "Invalid token"}, 403

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    
    # Extract user message and phone number (session key)
    entry = body["entry"][0]["changes"][0]["value"]
    message = entry["messages"][0]
    phone = message["from"]
    text = message["text"]["body"]
    
    # Get or create session state
    if phone not in sessions:
        sessions[phone] = {
            "messages": [], "intent": "",
            "lead_name": None, "lead_email": None,
            "lead_platform": None,
            "collecting_lead": False, "lead_captured": False, "response": ""
        }
    
    state = sessions[phone]
    state["messages"] = state["messages"] + [HumanMessage(content=text)]
    
    result = graph.invoke(state)
    sessions[phone].update(result)
    
    # Send reply back to WhatsApp
    await send_whatsapp_message(phone, result["response"])
    return {"status": "ok"}
```

**3. Send replies via Meta API**

```python
import httpx

async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
```

**4. Production considerations**
- Replace in-memory `sessions` dict with **Redis** for scalability and persistence across server restarts
- Use **ngrok** or deploy to a cloud provider (Railway, Render, AWS Lambda) to expose the webhook over HTTPS
- Add **rate limiting** and **signature verification** (`X-Hub-Signature-256` header) for security
- Store leads in a real CRM (HubSpot, Salesforce) instead of the mock function

---

## 📊 Evaluation Checklist

| Criteria | Implementation |
|---|---|
| Intent Detection | Keyword heuristics + LLM fallback in `utils/intent_classifier.py` |
| RAG Retrieval | TF-IDF style keyword scoring in `utils/rag_pipeline.py` |
| State Management | `AgentState` TypedDict across all turns in `agent/graph.py` |
| Tool Calling | `mock_lead_capture()` called only after all 3 fields collected |
| Code Clarity | Modular structure, docstrings, type hints throughout |
| Deployability | Webhook architecture documented above |

---

## 📄 License

MIT — Built for ServiceHive / Inflx Machine Learning Intern Assignment.
