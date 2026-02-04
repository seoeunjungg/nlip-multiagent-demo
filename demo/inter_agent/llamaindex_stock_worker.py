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
        d = msg.model_dump() if hasattr(msg, "model_dump") else {}
        print("[STOCK] RAW NLIP:", d, flush=True)
        print("[STOCK] format/subformat:", d.get("format"), d.get("subformat"), flush=True)
        print("[STOCK] content type:", type(d.get("content")), flush=True)
        fmt = (d.get("format") or d.get("Format") or "").lower()
        subfmt = (d.get("subformat") or d.get("Subformat") or "").lower()
        content = d.get("content") if "content" in d else d.get("Content")

        try:
            if fmt == "structured" and subfmt == "json" and isinstance(content, dict):
                tool = content.get("tool")
                args = content.get("args") or {}

                if tool == "get_stock_quote":
                    q = args.get("query", "")
                    out = await get_stock_quote(q)
                    return NLIP_Factory.create_text(out)

                return NLIP_Factory.create_text(f"❌ Unknown tool '{tool}' in structured NLIP request.")

            text = msg.extract_text()
            out = await get_stock_quote(text)
            return NLIP_Factory.create_text(out)

        except Exception as e:
            return NLIP_Factory.create_text(f"❌ Error: {str(e)}")

    async def stop(self):
        # No resources to clean up (LLM-free worker)
        return None


app = server.setup_server(LlamaIndexStockApplication())
