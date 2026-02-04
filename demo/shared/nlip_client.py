"""
NLIP client for inter-agent communication.
"""

import uuid
import httpx
from typing import Any, Optional

from nlip_sdk.nlip import NLIP_Factory
from nlip_sdk import nlip


class NLIPClient:
    """Client for sending NLIP messages to other agents."""

    def __init__(self, base_url: str, conversation_id: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.conversation_id = conversation_id or str(uuid.uuid4())

    def _with_conversation_token(self, message_dict: dict) -> dict:
        token_submsg = {
            "format": "token",
            "subformat": "conversation_coordinator",
            "content": self.conversation_id,
            "label": "conversation",
        }
        existing = message_dict.get("submessages") or []
        message_dict["submessages"] = [token_submsg] + existing
        return message_dict

    async def send_nlip(self, message: dict) -> nlip.NLIP_Message:
        """Send a raw NLIP JSON message dict and return the parsed NLIP_Message."""
        url = f"{self.base_url}/nlip/"
        payload = self._with_conversation_token(message)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60.0,
            )
            response.raise_for_status()
            return nlip.NLIP_Message.model_validate(response.json())

    async def send_text(self, content: str, subformat: str = "english") -> str:
        """Send NLIP text and return extracted text."""
        msg = NLIP_Factory.create_text(content)
        message_dict = msg.model_dump() if hasattr(msg, "model_dump") else msg
        message_dict["subformat"] = subformat
        resp = await self.send_nlip(message_dict)
        return resp.extract_text()

    async def send_tool_call(self, tool: str, args: dict[str, Any]) -> str:
        """
        NLIP-native agent<->agent RPC:
        format=structured, subformat=json, content={tool,args}
        """
        message = {
            "format": "structured",
            "subformat": "json",
            "content": {
                "tool": tool,
                "args": args,
            },
        }
        resp = await self.send_nlip(message)
        return resp.extract_text()
