import os
from contextlib import AsyncExitStack, asynccontextmanager

import json
import os

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client as http_client
import uvicorn
from fastapi import FastAPI

from fastapi.responses import JSONResponse
state: dict = {}
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

MAX_STEPS = 6
SYSTEM = (
    "Tu es un assistant spécialisé dans la reconnaissance de races de chats et "
    "de chiens (Oxford Pets). Utilise les outils pour classifier des images, "
    "trouver des images proches, filtrer et compter. "
    "Réponds dans la langue de l'utilisateur, de façon concise."
)
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

def result_text(result) -> str:
    parts = [c.text for c in result.content if getattr(c, "text", None)]
    return "\n".join(parts)


async def call_tool(name: str, args: dict) -> str:
    try:
        result = await state["session"].call_tool(name, args)
        return result_text(result)
    except Exception as e:
        return f"Erreur : {e}"


@app.get("/api/tools")
async def tools():
    return [{"name": t.name, "description": t.description} for t in state["tools"]]

@app.post("/api/chat")
async def chat(body: dict):
    """body = {"messages": [{"role": "user"|"assistant", "content": "..."}]}"""
    messages = [{"role": "system", "content": SYSTEM}] + body["messages"]
    trace = []
    model = body.get("model") or MODEL

    async with httpx.AsyncClient(timeout=180) as client:
        for _ in range(MAX_STEPS):
            r = await client.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "tools": state["ollama_tools"],
                    "stream": False,
                },
            )
            if r.status_code != 200:
                return JSONResponse(
                    {"detail": f"Erreur Ollama ({r.status_code}) : {r.text}"},
                    status_code=502,
                )
            msg = r.json()["message"]
            messages.append(msg)

            calls = msg.get("tool_calls")
            if not calls:
                return {"reply": msg.get("content", ""), "tools": trace}

            for call in calls:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                output = await call_tool(fn["name"], args)
                trace.append({"name": fn["name"], "args": args, "result": output})
                messages.append(
                    {"role": "tool", "tool_name": fn["name"], "content": output}
                )

    return {"reply": "Trop d'étapes d'outils, reformulez votre demande.", "tools": trace}

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