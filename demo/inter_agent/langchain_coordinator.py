"""
LangChain Coordinator Agent for Inter-Agent Communication Demo

This agent receives client requests and delegates tool execution to the LlamaIndex worker
agent via NLIP protocol. It demonstrates how different AI frameworks can collaborate.
"""

import os
from typing import Any
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler

# Import NLIP components
from nlip_sdk.nlip import NLIP_Factory
from nlip_sdk import nlip
from nlip_server import server

# Import shared utilities
from ..shared.nlip_client import NLIPClient

# Load environment variables
load_dotenv()

# Configuration
LLAMAINDEX_NEWS_URL = os.getenv("LLAMAINDEX_SERVER_URL", "http://localhost:8014")
LLAMAINDEX_STOCK_URL  = os.getenv("LLAMAINDEX_STOCK_URL",  "http://localhost:8013")

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)

@tool
async def get_stock_quote(query: str) -> str:
    """Get stock quote/price by delegating to the stock worker via NLIP."""
    client = NLIPClient(LLAMAINDEX_STOCK_URL)
    return await client.send_message(query.strip())

@tool
async def get_tech_news(topic: str, days: int = 1) -> str:
    """Get recent tech news about a topic by delegating to the LlamaIndex worker via NLIP."""
    print(f"\n🔄 [LangChain] Delegating tech news query for '{topic}' (days={days}) to LlamaIndex worker...")
    
    client = NLIPClient(LLAMAINDEX_NEWS_URL)
    query = f"Get recent tech news about '{topic}' in the last {days} days."
    
    response = await client.send_message(query)
    print(f"✅ [LangChain] Tech news response received from LlamaIndex worker\n")
    return response

class LangChainApplication(server.NLIP_Application):
    """LangChain application for inter-agent communication."""
    
    async def startup(self):
        print("🚀 Starting LangChain Agent...")
        print("This agent delegates tool execution to LlamaIndex worker via NLIP protocol")

    async def shutdown(self):
        print("🛑 Shutting down LangChain Agent")
        return None

    async def create_session(self) -> server.NLIP_Session:
        return LangChainSession()


class LangChainSession(server.NLIP_Session):
    """Chat session using LangChain with tool delegation."""
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self.agent_executor = None
        self.tools = []

    async def start(self):
        """Initialize LangChain components for coordination."""
        try:
            print("🔧 Initializing LangChain components...")

            self.llm = ChatOpenAI(
                model="gpt-4o-mini",         
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3,
                max_tokens=1200,
                callbacks=[StreamingCallbackHandler()],
            )

            # Define available tools (all delegating to LlamaIndex server)
            self.tools = [
                get_tech_news,
                get_stock_quote
            ]
            
            # Create prompt template for coordination
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                "You are a helpful assistant that answers questions about technology companies. "
                "Use tools to fetch up-to-date stock prices or recent news when needed. "
                "Answer directly when the question can be answered without external data."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # Create agent
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
            
            print("✅ LangChain components initialized successfully.")
            print(f"🛠️ Available delegating tools: {[tool.name for tool in self.tools]}")
            print(f"🌐 LlamaIndex worker server URL: {LLAMAINDEX_NEWS_URL}")
            
        except Exception as e:
            print(f"❌ Error initializing LangChain components: {e}")
            raise

    async def execute(self, msg: nlip.NLIP_Message) -> nlip.NLIP_Message:
        """Execute user query using LangChain agent with dynamic tool use."""
        logger = self.get_logger()
        text = msg.extract_text()
        logger.info(f"\n📨 [LangChain] Processing client query: {text}\n" + "=" * 80 + "\n")

        try:
            result = await self.agent_executor.ainvoke({"input": text})

            output = result.get("output") if isinstance(result, dict) else str(result)

            return NLIP_Factory.create_text(output)

        except Exception as e:
            logger.exception("Exception in LangChain execution")
            return NLIP_Factory.create_text(f"❌ Error: {str(e)}")


    async def stop(self):
        """Clean up resources."""
        print("🛑 Stopping LangChain session")
        self.llm = None
        self.agent_executor = None
        self.tools = []

app = server.setup_server(LangChainApplication())