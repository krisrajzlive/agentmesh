"""A tiny stdio MCP server used to test the bridge."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("calc")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def shout(text: str) -> str:
    """Upper-case some text."""
    return text.upper()


@server.tool()
def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


if __name__ == "__main__":
    server.run("stdio")
