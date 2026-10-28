import json

import httpx
from mcp import ClientSession

from config import MAX_STEPS, MODEL, OLLAMA_URL, SYSTEM
from mcp_tools import call_tool


class OllamaError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def run_agent(
    session: ClientSession,
    ollama_tools: list[dict],
    messages: list[dict],
    model: str | None = None,
    on_tool=None,
    options: dict | None = None,
) -> tuple[str, list[dict]]:
    """Renvoie (réponse, trace des outils appelés).

    on_tool : callback optionnel appelé après chaque appel d'outil.
    options : options Ollama (ex. {"temperature": 0} pour l'évaluation).
    """
    model = model or MODEL
    messages = [{"role": "system", "content": SYSTEM}, *messages]
    trace: list[dict] = []

    async with httpx.AsyncClient(timeout=180) as client:
        for _ in range(MAX_STEPS):
            r = await client.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "tools": ollama_tools,
                    "stream": False,
                    **({"options": options} if options else {}),
                },
            )
            if r.status_code != 200:
                raise OllamaError(
                    r.status_code, f"Erreur Ollama ({r.status_code}) : {r.text}"
                )
            msg = r.json()["message"]
            messages.append(msg)

            calls = msg.get("tool_calls")
            if not calls:
                return msg.get("content", ""), trace

            for call in calls:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                output = await call_tool(session, fn["name"], args)
                entry = {"name": fn["name"], "args": args, "result": output}
                trace.append(entry)
                if on_tool:
                    on_tool(entry)
                messages.append(
                    {"role": "tool", "tool_name": fn["name"], "content": output}
                )

    return "Trop d'étapes d'outils, reformulez votre demande.", trace
