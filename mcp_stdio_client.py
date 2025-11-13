#!/usr/bin/env python3
"""
MCP Stdio Client
Handles stdio communication with mcp_medical_server.py using MCP protocol
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("pip install mcp")
    sys.exit(1)

logger = logging.getLogger(__name__)


class MCPStdioClient:
    """Client for communicating with MCP medical server via stdio protocol."""
    
    def __init__(self, server_path: str = "./mcp_medical_server.py"):
        """
        Initialize MCP stdio client.
        
        Args:
            server_path: Path to the MCP server script
        """
        self.server_path = Path(server_path).resolve()
        if not self.server_path.exists():
            raise FileNotFoundError(f"MCP server not found at: {self.server_path}")
        
        self.session: Optional[ClientSession] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._stdio_context = None
        
    async def connect(self):
        """Connect to MCP server via stdio."""
        try:
            logger.info(f"Connecting to MCP server at: {self.server_path}")
            
            # Create server parameters
            server_params = StdioServerParameters(
                command="python3",
                args=[str(self.server_path)],
                env=None
            )
            
            # Create stdio client connection
            # stdio_client returns an async context manager that we need to enter
            self._stdio_context = stdio_client(server_params)
            
            # Enter the async context manager properly
            # __aenter__() returns an awaitable that yields (read_stream, write_stream)
            read_stream, write_stream = await self._stdio_context.__aenter__()
            
            # Create client session
            self.session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            await self.session.initialize()
            
            logger.info("✓ MCP client connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Clean up on error
            if hasattr(self, '_stdio_context') and self._stdio_context:
                try:
                    await self._stdio_context.__aexit__(None, None, None)
                except:
                    pass
                self._stdio_context = None
            raise
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        try:
            # Exit the stdio context manager if it exists
            if hasattr(self, '_stdio_context') and self._stdio_context:
                try:
                    await self._stdio_context.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error exiting stdio context: {e}")
                finally:
                    self._stdio_context = None
            
            # Close session if it exists
            if self.session:
                try:
                    # ClientSession doesn't have __aexit__, just close streams if needed
                    logger.info("MCP client disconnected")
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")
                finally:
                    self.session = None
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result as dictionary
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect() first.")
        
        try:
            logger.debug(f"Calling MCP tool: {tool_name} with args: {arguments}")
            
            # Call the tool
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract text content from result
            if result and result.content:
                # MCP returns TextContent objects
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                    elif isinstance(content_item, dict) and 'text' in content_item:
                        text_content += content_item['text']
                    elif isinstance(content_item, str):
                        text_content += content_item
                
                # The MCP server returns formatted text, try to extract structured data
                # For search_medical_knowledge, the server returns formatted text with results
                # We need to parse it or return as-is for the LLM to process
                
                # Try to parse as JSON if possible
                try:
                    parsed = json.loads(text_content)
                    return parsed
                except json.JSONDecodeError:
                    # For search results, the text contains formatted results
                    # Return as structured dict that can be parsed by format_mcp_results
                    return {
                        "text": text_content,
                        "tool": tool_name,
                        "arguments": arguments,
                        "raw_response": text_content
                    }
            else:
                return {
                    "error": "No content returned",
                    "tool": tool_name,
                    "results": []
                }
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "error": str(e),
                "tool": tool_name,
                "arguments": arguments
            }
    
    async def search_medical_knowledge(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search medical knowledge base."""
        return await self.call_tool(
            "search_medical_knowledge",
            {"query": query, "top_k": top_k}
        )
    
    async def search_by_condition(self, condition: str) -> Dict[str, Any]:
        """Search by medical condition."""
        return await self.call_tool(
            "search_by_condition",
            {"condition": condition}
        )
    
    async def search_by_treatment(self, treatment: str) -> Dict[str, Any]:
        """Search by treatment."""
        return await self.call_tool(
            "search_by_treatment",
            {"treatment": treatment}
        )
    
    async def search_by_symptom(self, symptom: str) -> Dict[str, Any]:
        """Search by symptom."""
        return await self.call_tool(
            "search_by_symptom",
            {"symptom": symptom}
        )
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information."""
        return await self.call_tool("get_database_info", {})
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Test function
async def test_mcp_client():
    """Test MCP client connection and tool calls."""
    client = MCPStdioClient()
    try:
        await client.connect()
        
        # Test search
        result = await client.search_medical_knowledge("knee pain", top_k=3)
        print("Test result:", json.dumps(result, indent=2))
        
    finally:
        await client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_mcp_client())

