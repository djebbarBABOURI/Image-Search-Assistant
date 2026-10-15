from mcp import ClientSession


def to_ollama_tools(mcp_tools) -> list[dict]:
    """Convertit les outils MCP en définitions de fonctions pour Ollama."""
    return [
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
