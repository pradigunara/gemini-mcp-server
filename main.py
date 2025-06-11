#!/usr/bin/env python3
"""Entry point for Gemini MCP Server"""
import asyncio
from server import main

def cli():
    """CLI entry point"""
    asyncio.run(main())

if __name__ == "__main__":
    cli()