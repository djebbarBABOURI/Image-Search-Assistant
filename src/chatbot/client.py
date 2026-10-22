import base64
import json
import re
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client as http_client

from agent import OllamaError, run_agent
from config import MCP_URL, OLLAMA_HOST, MODEL
from mcp_tools import HIDDEN, to_ollama_tools

STATIC_DIR = Path(__file__).parent / "static"
IMAGE_ID = re.compile(r"^[\w.\- ]{1,120}$")

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
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def result_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "text", None))


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/tools")
async def tools():
    return [
        {"name": t.name, "description": t.description}
        for t in state["tools"]
        if t.name not in HIDDEN
    ]


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


@app.get("/api/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
        return {"models": names, "default": MODEL}
    except Exception:
        return {"models": [], "default": MODEL}


@app.get("/api/image/{image_id}")
async def image(image_id: str, size: str = "thumb"):
    if not IMAGE_ID.match(image_id) or size not in ("thumb", "full"):
        return JSONResponse({"detail": "Paramètres invalides"}, status_code=400)

    try:
        res = await state["session"].call_tool(
            "get_image", {"image_id": image_id, "size": size}
        )
        data = json.loads(result_text(res))
        content = base64.b64decode(data["data_base64"])
    except Exception:
        return JSONResponse({"detail": "Image introuvable"}, status_code=404)

    return Response(
        content,
        media_type=data.get("mime", "image/jpeg"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/api/upload")
async def upload(request: Request):
    raw = await request.body()
    if not raw or len(raw) > 10_000_000:
        return JSONResponse(
            {"detail": "Image vide ou supérieure à 10 Mo"}, status_code=400
        )

    res = await state["session"].call_tool(
        "register_image",
        {"filename": "upload", "data_base64": base64.b64encode(raw).decode()},
    )

    try:
        data = json.loads(result_text(res))
    except json.JSONDecodeError:
        data = {}

    if "image_id" not in data:
        return JSONResponse(
            {"detail": data.get("error", "Échec de l'envoi")}, status_code=400
        )
    return data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)