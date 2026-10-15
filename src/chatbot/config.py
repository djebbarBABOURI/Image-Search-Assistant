import os

MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

MAX_STEPS = 6
SYSTEM = (
    "Tu es un assistant spécialisé dans la reconnaissance de races de chats et "
    "de chiens (Oxford Pets). Utilise les outils pour classifier des images, "
    "trouver des images proches, filtrer et compter. "
    "Réponds dans la langue de l'utilisateur, de façon concise."
)
