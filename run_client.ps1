Set-Location $PSScriptRoot

uv run --with-requirements requirements.txt --with fastapi --with uvicorn --with httpx python src/chatbot/client.py