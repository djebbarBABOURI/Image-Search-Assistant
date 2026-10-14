from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client as http_client

from agent import OllamaError, run_agent
from config import MCP_URL
from mcp_tools import to_ollama_tools

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        streams = await stack.enter_async_context(http_client(MCP_URL))
        session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await session.initialize()
        tools = (await session.list_tools()).tools
        state["session"] = session
        state["tools"] = tools
        state["ollama_tools"] = to_ollama_tools(tools)
        print(f"MCP connecté : {[t.name for t in tools]}")
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/tools")
async def tools():
    return [{"name": t.name, "description": t.description} for t in state["tools"]]


@app.post("/api/chat")
async def chat(body: dict):
    """body = {"messages": [{"role": "user"|"assistant", "content": "..."}]}"""
    try:
        reply, trace = await run_agent(
            state["session"],
            state["ollama_tools"],
            body["messages"],
            body.get("model"),
        )
    except OllamaError as e:
        return JSONResponse({"detail": e.detail}, status_code=502)
    return {"reply": reply, "tools": trace}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
