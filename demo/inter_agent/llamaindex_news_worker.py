"""
LlamaIndex Agent for Inter-Agent Communication Demo

This agent executes the actual tech news tool implementations and serves
requests delegated from the LangChain agent via NLIP protocol.
"""

import os
from dotenv import load_dotenv

from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from nlip_sdk.nlip import NLIP_Factory
from nlip_sdk import nlip
from nlip_server import server

from ..shared.news_tools import get_tech_news_brief

load_dotenv()


class LlamaIndexApplication(server.NLIP_Application):
    """LlamaIndex application for inter-agent communication."""
    
    async def startup(self):
        print("🔧 Starting LlamaIndex Agent...")
        print("This agent executes tech news tools for requests delegated via NLIP protocol")

    async def shutdown(self):
        print("🛑 Shutting down LlamaIndex Agent")
        return None

    async def create_session(self) -> server.NLIP_Session:
        return LlamaIndexSession()


class LlamaIndexSession(server.NLIP_Session):
    """Chat session using LlamaIndex with actual tool implementations."""
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self.agent = None
        self.tools = []
        self.context = None

    async def start(self):
        """Initialize LlamaIndex components for tool execution."""
        try:
            print("🔧 Initializing LlamaIndex components...")

            self.llm = OpenAI(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3,
                max_tokens=512,
            )
            
            self.tools = [
                FunctionTool.from_defaults(
                    fn=get_tech_news_brief,
                    name="get_tech_news_brief",
                    description=(
                        "Fetch recent technology/AI/cybersecurity news about a topic in the last few days. "
                        "Args: topic (str), days (int, optional)."
                    ),
                ),
            ]
            
            self.agent = FunctionAgent(
                tools=self.tools,
                llm=self.llm,
                verbose=True,
                system_prompt=(
                    "You are a specialized tech news agent. You fetch and summarize recent news about "
                    "computer science, AI, and cybersecurity. You have a tool that retrieves recent "
                    "articles (titles, sources, publication dates, short descriptions, URLs). "
                    "Use that tool to collect information, then synthesize concise, factual summaries. "
                    "Do not invent sources; if information is missing or uncertain, say so explicitly."
                )
            )

            self.context = Context(self.agent)
            
            print("✅ LlamaIndex components initialized successfully.")
            print(f"🛠️ Available tech news tools: {[tool.metadata.name for tool in self.tools]}")
            
        except Exception as e:
            print(f"❌ Error initializing LlamaIndex components: {e}")
            raise

    async def execute(self, msg: nlip.NLIP_Message) -> nlip.NLIP_Message:
        logger = self.get_logger()

        d = msg.model_dump() if hasattr(msg, "model_dump") else {}
        print("[NEWS] RAW NLIP:", d, flush=True)
        print("[NEWS] format/subformat:", d.get("format"), d.get("subformat"), flush=True)
        print("[NEWS] content type:", type(d.get("content")), flush=True)
        fmt = (d.get("format") or d.get("Format") or "").lower()
        subfmt = (d.get("subformat") or d.get("Subformat") or "").lower()
        content = d.get("content") if "content" in d else d.get("Content")

        try:
            if fmt == "structured" and subfmt == "json" and isinstance(content, dict):
                tool = content.get("tool")
                args = content.get("args") or {}

                if tool == "get_tech_news_brief":
                    topic = args.get("topic", "")
                    days = int(args.get("days", 1))
                    out = await get_tech_news_brief(topic=topic, days=days)
                    return NLIP_Factory.create_text(out)

                return NLIP_Factory.create_text(f"❌ Unknown tool '{tool}' in structured NLIP request.")

            text = msg.extract_text()
            print(f"\n🔧 [LlamaIndex] Processing delegated query: {text}")
            print("=" * 80)

            self.context = Context(self.agent)
            response = await self.agent.run(text, ctx=self.context)
            response_text = str(response)

            print("=" * 80)
            print(f"✅ [LlamaIndex] Completed processing, returning result to coordinator\n")
            logger.info(f"LlamaIndex Response: {response_text}")
            return NLIP_Factory.create_text(response_text)

        except Exception as e:
            logger.error(f"Exception in LlamaIndex execution: {e}")
            return NLIP_Factory.create_text(f"❌ Error processing request: {str(e)}")

    async def stop(self):
        """Clean up resources."""
        print("🛑 Stopping LlamaIndex worker session")
        self.llm = None
        self.agent = None
        self.tools = []
        self.context = None

app = server.setup_server(LlamaIndexApplication())
