# Integrating NLIP with Agent Development Frameworks

Agent development frameworks such as LangChain, AutoGen, and LlamaIndex have significantly simplified the construction of large language model (LLM)–based applications. However, this growing diversity has resulted in a fragmented ecosystem in which agents lack a shared communication protocol, limiting interoperability across domains and across development frameworks. To address this challenge, we demonstrate the integration of the Natural Language Interaction Protocol (NLIP) with agent development frameworks to enable seamless, standardized communication among heterogeneous agents.

NLIP operates as an external communication layer that encapsulates standalone agent applications or bridges agents implemented using different development frameworks. It provides a non-intrusive interface for message passing and task delegation without requiring changes to an agent’s internal logic or the framework’s tool-calling mechanisms. Using NLIP, agents developed under different frameworks can interact, exchange information, and coordinate tasks across domains, supporting more flexible and interconnected multi-agent systems.

In this project, the approach is demonstrated through a dynamic multi-agent information and analysis task. A coordinator agent built with LangChain receives user queries and reasons about how to decompose them. Depending on the query, the coordinator selectively delegates subtasks to specialized LlamaIndex worker agents via NLIP. These workers retrieve recent news information and current stock market data from external sources. The coordinator then synthesizes the results and, when requested, generates scenario-based stock outlooks grounded in real-time data. All inter-agent interactions are conducted using NLIP messages, enabling clean separation between reasoning and execution.

Overall, this work shows that NLIP integrates seamlessly with existing agent development frameworks and establishes a foundation for scalable, interoperable, and dynamically coordinated multi-agent systems.

Demonstrates NLIP integration with AI agent frameworks, showcasing dynamic inter-agent communication and reasoning-driven task delegation.

📹 **[Watch Demo Video](https://drive.google.com/file/d/1f1CsS9Om2vA6Il5vjyZ2o9MNMnO1cwoc/view?usp=sharing)** - See the demo in action!

## 🎯 What This Shows

1. **🔄 Inter-Agent Communication**: A LangChain-based coordinator dynamically delegates tasks to LlamaIndex-based worker agents via NLIP.
2. **🧠 Dynamic Reasoning and Delegation**: The coordinator determines at runtime whether to answer directly, retrieve news, fetch stock data, or combine multiple agents’ outputs.

**Key Capabilities:**
- Cross-framework communication via the NLIP protocol
- Dynamic tool and agent selection (no fixed intent routing)
- Real-time data grounding using external news and stock sources
- Scenario-based, uncertainty-aware stock outlook generation

## 🏗️ Project Structure

```
nlip-multiagent-news-demo/
├── README.md
├── pyproject.toml
├── .env
├── demo/
│   ├── inter_agent/
│   │   ├── langchain_coordinator.py      # Reasoning + orchestration agent
│   │   ├── llamaindex_news_worker.py     # News retrieval worker
│   │   ├── llamaindex_stock_worker.py    # Stock data worker (no LLM)
│   ├── shared/
│   │   ├── news_tools.py
│   │   ├── stock_tools.py
│   │   ├── nlip_client.py
│   │   └── __init__.py
```

## 🚀 Quick Start

**Prerequisites:** Python 3.10+, Poetry (recommended)

**Setup:**
```bash
git clone <repository-url>
cd nlip-multiagent-news-demo

# Clone the nlip-server dependency into the project directory
git clone https://github.com/seoeunjungg/nlip-multiagent-news-demo.git

# Install project dependencies
poetry install  # or: pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env with: # Configure required API keys (e.g., OPENAI_API_KEY, NEWS_API_KEY / SERPER_API_KEY)
```

### Run Inter-Agent Demo

Open four terminals in the project directory:

**Terminal 1 - Start Stock Worker:**
```bash
poetry run uvicorn demo.inter_agent.llamaindex_stock_worker:app \
  --host 0.0.0.0 --port 8013
```

**Terminal 2 - Start News Worker:**
```bash
poetry run uvicorn demo.inter_agent.llamaindex_news_worker:app \
  --host 0.0.0.0 --port 8014
```
**Terminal 3 - Start LangChain Coordinator:**
```bash
poetry run uvicorn demo.inter_agent.langchain_coordinator:app \
  --host 0.0.0.0 --port 8012
```

**Terminal 4 - NLIP client:**
```bash
curl -X POST http://localhost:8012/nlip/ \
  -H "Content-Type: application/json" \
  -d '{"format":"text","subformat":"english","content":"Predict NVDA’s stock outlook over the next 2 weeks using current price and recent news."}'
```

## 📊 Demo Scenarios

**Inter-Agent:** Client → LangChain Coordinator → (NLIP) → LlamaIndex Workers → External Data Sources


## 📋 Dependencies

This demo requires:
- **Python 3.10+** (required for NLIP SDK)
- **NLIP SDK & Server** (installed from PyPI: `nlip-sdk>=0.1.2`, `nlip-server`)
- **OPENAI_API_KEY** (for AI model access)
- **NEWS_API_KEY** (optional: for news worker when Serper is not available)
- **SERPER_API_KEY** (for broader search)
