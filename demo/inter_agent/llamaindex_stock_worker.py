"""
LlamaIndex Stock Worker (Tool Microservice)

This worker is intentionally LLM-free. It receives a ticker/company query via NLIP
and returns OHLCV data using the shared stock tool implementation.
"""

from dotenv import load_dotenv

from nlip_sdk.nlip import NLIP_Factory
from nlip_sdk import nlip
from nlip_server import server

from ..shared.stock_tools import get_stock_quote

load_dotenv()


class LlamaIndexStockApplication(server.NLIP_Application):
    async def startup(self):
        print("📈 Starting Stock Worker (no LLM)...")

    async def shutdown(self):
        print("🛑 Shutting down Stock Worker")
        return None

    async def create_session(self) -> server.NLIP_Session:
        return LlamaIndexStockSession()


class LlamaIndexStockSession(server.NLIP_Session):
    async def start(self):
        print("✅ Stock worker initialized.")

    async def execute(self, msg: nlip.NLIP_Message) -> nlip.NLIP_Message:
        text = msg.extract_text()
        try:
            out = await get_stock_quote(text)
            return NLIP_Factory.create_text(out)
        except Exception as e:
            return NLIP_Factory.create_text(f"❌ Error: {str(e)}")

    async def stop(self):
        # No resources to clean up (LLM-free worker)
        return None


app = server.setup_server(LlamaIndexStockApplication())
