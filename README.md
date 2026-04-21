# AutoStream · Social-to-Lead AI Agent
### Machine Learning Intern Assignment — ServiceHive / Inflx

> A production-grade conversational AI agent that converts user intent into qualified business leads using LangGraph, RAG, and tool execution.

---

## ✨ Key Features

- 💬 Conversational AI agent with memory
- 📚 Retrieval-Augmented Generation (RAG)
- 🎯 Intent detection (general vs high-intent)
- 🧰 Tool execution (lead capture)
- ⚡ Demo-safe fallback mode (no API required)
- 🔄 Stateful workflow using LangGraph

---

## 📁 Project Structure
autostream-agent/
├── main.py # CLI entrypoint
├── requirements.txt
├── README.md
├── knowledge_base/
│ └── autostream_kb.json # Local knowledge base
├── agent/
│ └── graph.py # LangGraph workflow
├── tools/
│ └── lead_capture.py # Lead capture tool
└── utils/
├── rag_pipeline.py # RAG retrieval logic
└── intent_classifier.py # Intent classification


---

## 🚀 How to Run Locally

### ✅ Prerequisites
- Python 3.9 or higher
- (Optional) OpenAI API key for real LLM responses

---

### 1. Clone and setup

```bash
git clone https://github.com/sejalmohanwankhade/Social-to-Lead-Agentic-Workflow.git
cd autostream-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

2. (Optional) Set API key
export OPENAI_API_KEY=your-key

# Windows PowerShell:
# $env:OPENAI_API_KEY="your-key"

⚡ Demo Mode (No API Key Required)

If no API key is provided, the agent runs in demo mode using fallback logic.

This ensures:

App never crashes
Intent detection works
Lead capture flow works

⚠ This mode is used for demonstration purposes.

3. Run the agent
python main.py
🎥 Demo Instructions

Try these queries:

What features do you offer?
I want pricing
I want to sign up

Then provide:

Name
Email
Platform

👉 The agent will capture the lead successfully.

💬 Example Interaction
You: What features do you offer?
Aria: We offer video editing automation, analytics, and integrations.

You: I want pricing
Aria: Our pricing starts at $29/month. Would you like to book a demo?

You: Yes, I want to sign up
Aria: Great! Can I get your name and email?

You: Sejal
You: sejal@email.com

Aria: 🎉 Lead captured successfully!
🏗 Architecture Overview
User Input
     ↓
Intent Detection
     ↓
RAG Retrieval (Knowledge Base)
     ↓
LLM / Fallback Logic
     ↓
Tool Execution (Lead Capture)
🧠 Design Decisions
Why LangGraph?

LangGraph enables deterministic state-driven workflows, making it ideal for structured flows like lead capture.
Unlike autonomous agents, this system ensures predictable transitions.

State Management

The agent uses a shared AgentState to track:

Conversation history
User intent
Lead details (name, email, platform)
Workflow state (collecting vs completed)
RAG System
Uses local JSON knowledge base
Retrieves relevant content based on query
Improves response accuracy
Tool Execution

Lead capture is implemented as a tool:

capture_lead(name, email, platform)
Triggered only when:

User shows high intent
Required details are collected
📊 Evaluation Checklist
Criteria	Implementation
Intent Detection	Keyword-based classifier
RAG Retrieval	Local knowledge base
State Management	LangGraph state
Tool Execution	Lead capture function
Code Structure	Modular and clean
Demo Stability	Fallback mode enabled
🚀 Future Improvements
Streamlit UI (chat interface)
CRM integration (HubSpot, Salesforce)
WhatsApp / API deployment
Advanced LLM routing
📄 License

MIT — Built for ServiceHive / Inflx ML Intern Assignment