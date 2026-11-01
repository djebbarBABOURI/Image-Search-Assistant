import re

from mcp import ClientSession
 
URI_RE = re.compile(r"pets://[\w\-./]+")
UNKNOWN = ("Image inconnue", "Race inconnue")

HIDDEN = {"register_image", "get_image"}


def to_ollama_tools(tools):
    return [
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
        if t.name not in HIDDEN
    ]


def result_text(result) -> str:
    """Extrait le texte d'un résultat d'outil MCP."""
    parts = [c.text for c in result.content if getattr(c, "text", None)]
    return "\n".join(parts)


async def call_tool(session: ClientSession, name: str, args: dict) -> str:
    """Appelle un outil MCP ; renvoie le texte, ou un message d'erreur."""
    try:
        return result_text(await session.call_tool(name, args))
    except Exception as e:
        return f"Erreur : {e}"


def attr(obj, *names, default=None):
    """Premier attribut présent (selon la version du SDK : camelCase ou snake_case)."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def prompt_text(result) -> str:
    """Texte d'un résultat get_prompt (messages concaténés)."""
    parts = [getattr(m.content, "text", None) for m in result.messages]
    return "\n\n".join(p for p in parts if p)


async def read_resource(session: ClientSession, uri: str) -> str:
    """Lit une ressource MCP et renvoie son texte."""
    result = await session.read_resource(uri)
    return "\n".join(c.text for c in result.contents if getattr(c, "text", None))


async def resolve_resources(session: ClientSession, text: str) -> list[dict]:
    """Lit les ressources citées dans un texte (ex. un prompt MCP).

    Le LLM n'a pas d'outil pour lire une ressource : on les lui fournit
    directement. Les ressources introuvables sont ignorées.
    """
    uris = dict.fromkeys(u.rstrip(".,;:") for u in URI_RE.findall(text))
    found = []
    for uri in uris:
        try:
            content = await read_resource(session, uri)
        except Exception:
            continue
        if content and not content.startswith(UNKNOWN):
            found.append({"uri": uri, "text": content})
    return found