import os
from contextlib import AsyncExitStack, asynccontextmanager

import json
import os

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client as http_client
import uvicorn
from fastapi import FastAPI

state: dict = {}
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        streams = await stack.enter_async_context(http_client(MCP_URL))
        session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()
        tools = (await session.list_tools()).tools
        state["session"] = session
        state["tools"] = tools
        state["ollama_tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": getattr(t, "input_schema", None)
                    or getattr(t, "inputSchema", None)
                    or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]
        print(f"MCP connecté : {[t.name for t in tools]}")
        yield


app = FastAPI(lifespan=lifespan)

async def ask(session: ClientSession, tools: list[dict], question: str) -> str:
    messages = [{"role": "user", "content": question}]

    async with httpx.AsyncClient(timeout=180) as client:
        while True:
            r = await client.post(
                OLLAMA_URL,
                json={"model": MODEL, "messages": messages, "tools": tools, "stream": False},
            )
            r.raise_for_status()
            msg = r.json()["message"]
            messages.append(msg)

            calls = msg.get("tool_calls")
            if not calls:
                return msg.get("content", "")

            for call in calls:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                print(f"[outil] {fn['name']}({args})")

                result = await session.call_tool(fn["name"], args)
                text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
                messages.append({"role": "tool", "tool_name": fn["name"], "content": text})


async def main():
    async with http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": getattr(t, "inputSchema", None)
                        or getattr(t, "input_schema", None)
                        or {"type": "object", "properties": {}},
                    },
                }
                for t in mcp_tools
            ]
            print("Outils :", [t.name for t in mcp_tools])

            reply = await ask(session, tools, "Combien d'images sont prédites boxer ?")
            print("\nRéponse :", reply)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)