import asyncio

from mcp import ClientSession

from mcp.client.streamable_http import streamable_http_client as http_client

MCP_URL = "http://localhost:8000/mcp"


async def main():
    async with http_client(MCP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()

            # Lister les outils
            tools = (await session.list_tools()).tools
            print([t.name for t in tools])

            # Appeler un outil (exemple avec list_breeds, sans paramètre)
            result = await session.call_tool("list_breeds", {})
            print("\n".join(c.text for c in result.content if getattr(c, "text", None)))


asyncio.run(main())