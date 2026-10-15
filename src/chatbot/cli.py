import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client as http_client

from agent import run_agent
from config import MCP_URL
from mcp_tools import to_ollama_tools

QUESTION = "Combien d'images sont prédites boxer ?"


async def main():
    async with http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            print("Outils :", [t.name for t in mcp_tools])

            reply, _ = await run_agent(
                session,
                to_ollama_tools(mcp_tools),
                [{"role": "user", "content": QUESTION}],
                on_tool=lambda e: print(f"[outil] {e['name']}({e['args']})"),
            )
            print("\nRéponse :", reply)


if __name__ == "__main__":
    asyncio.run(main())
