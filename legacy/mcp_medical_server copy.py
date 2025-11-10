#!/usr/bin/env python3
"""
Medical Knowledge MCP Server
A Model Context Protocol server for medical knowledge retrieval using ChromaDB
Compatible with the existing Langapproach.py Flask server
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple
import asyncio
import time

# Add error handling for missing packages
try:
    import dotenv
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("pip install mcp chromadb sentence-transformers python-dotenv")
    sys.exit(1)

# Load environment variables
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration - MUST MATCH YOUR Langapproach.py settings
CHROMA_DB_PATH = "./chroma_db_collective"  # Updated database path
CHROMA_COLLECTION_NAME = "books_collection"  # Updated collection name
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Updated embedding model
CACHE_TTL_SECONDS = 3600  # Cache TTL: 1 hour

# ---- TTL Cache Implementation ----
class TTLCache:
    """Simple TTL cache implementation to prevent memory bloat."""
    
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                # Expired, remove from cache
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        self._cache[key] = (value, time.time())
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

# ---- Medical Database Class ----
class MedicalDatabase:
    """Medical knowledge database using ChromaDB - shared with Langapproach.py."""
    
    def __init__(self, chroma_db_path: str):
        self._chroma_db_path = chroma_db_path
        self._chroma_client = None
        self._collection = None
        self._search_cache = TTLCache(CACHE_TTL_SECONDS)
        self._init_database()
    
    def _init_database(self):
        """Initialize ChromaDB and embeddings."""
        logger.info("=" * 60)
        logger.info("Initializing Medical Knowledge MCP Server...")
        logger.info("=" * 60)
        
        try:
            # Note: BGE embeddings don't require API keys - they run locally
            logger.info(f"Initializing BGE embeddings (model: {EMBEDDING_MODEL})...")
            logger.info("✓ BGE embeddings will be handled by ChromaDB embedding function")
            
            # Initialize Chroma client
            logger.info(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
            self._chroma_client = chromadb.PersistentClient(path=self._chroma_db_path)
            logger.info("✓ ChromaDB client connected")
            
            # Get embedding function
            logger.info("Setting up BGE embedding function...")
            bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
            logger.info("✓ BGE embedding function ready")
            
            # Get collection
            logger.info(f"Loading collection: {CHROMA_COLLECTION_NAME}")
            self._collection = self._chroma_client.get_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=bge_ef
            )
            
            count = self._collection.count()
            logger.info("=" * 60)
            logger.info(f"✓ SUCCESS: Database loaded with {count:,} documents")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"ERROR: Database initialization failed!")
            logger.error(f"Error: {e}")
            logger.error(f"ChromaDB path: {self._chroma_db_path}")
            logger.error(f"Collection name: {CHROMA_COLLECTION_NAME}")
            logger.error("=" * 60)
            raise
    
    
    def search_knowledge(self, query: str, top_k: int = 5) -> dict:
        """Search medical knowledge base."""
        logger.info(f"SEARCH: '{query}' (top_k={top_k})")
        
        # Check cache
        cache_key = f"{query}_{top_k}"
        cached_result = self._search_cache.get(cache_key)
        if cached_result is not None:
            logger.info("⚡ Using cached result")
            return cached_result
        
        try:
            # Query ChromaDB
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Debug: Log metadata structure
            if metadatas and len(metadatas) > 0:
                logger.info(f"Sample metadata structure: {metadatas[0]}")
                logger.info(f"Metadata keys: {list(metadatas[0].keys()) if metadatas[0] else 'No metadata'}")
            
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                # Convert distance to cosine similarity
                # ChromaDB uses cosine distance = 1 - cosine_similarity
                # So: cosine_similarity = 1 - distance
                cosine_similarity = 1 - distance
                
                # Ensure similarity is between 0 and 1
                cosine_similarity = max(0, min(1, cosine_similarity))
                
                # Handle different metadata structures
                filename = metadata.get('filename', metadata.get('source', metadata.get('file', 'unknown')))
                chunk_id = metadata.get('chunk_id', metadata.get('chunk', metadata.get('id', i)))
                
                search_results.append({
                    "rank": i + 1,
                    "content": doc,
                    "source": f"{filename}_chunk_{chunk_id}",
                    "similarity": float(cosine_similarity),
                    "relevance_score": f"{cosine_similarity:.2%}"
                })
            
            result = {
                "query": query,
                "results_count": len(search_results),
                "results": search_results
            }
            
            # Cache result
            self._search_cache.set(cache_key, result)
            logger.info(f"✓ Found {len(search_results)} results")
            
            return result
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "query": query,
                "results_count": 0,
                "results": [],
                "error": str(e)
            }
    
    def search_by_condition(self, condition: str) -> dict:
        """Search for medical condition information."""
        logger.info(f"CONDITION SEARCH: '{condition}'")
        enhanced_query = f"medical condition {condition} symptoms diagnosis treatment causes"
        return self.search_knowledge(enhanced_query, top_k=3)
    
    def search_by_treatment(self, treatment: str) -> dict:
        """Search for treatment information."""
        logger.info(f"TREATMENT SEARCH: '{treatment}'")
        enhanced_query = f"treatment therapy {treatment} protocol procedure medication"
        return self.search_knowledge(enhanced_query, top_k=3)
    
    def search_by_symptom(self, symptom: str) -> dict:
        """Search by symptom."""
        logger.info(f"SYMPTOM SEARCH: '{symptom}'")
        enhanced_query = f"symptom {symptom} causes conditions diagnosis"
        return self.search_knowledge(enhanced_query, top_k=3)
    
    def get_statistics(self) -> dict:
        """Get database statistics."""
        try:
            count = self._collection.count()
            # Clean expired cache entries
            expired_count = self._search_cache.clear_expired()
            if expired_count > 0:
                logger.info(f"Cleaned {expired_count} expired cache entries")
            
            return {
                "total_documents": count,
                "collection_name": CHROMA_COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL,
                "database_path": CHROMA_DB_PATH,
                "cache_size": self._search_cache.size(),
                "status": "operational"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# ---- MCP Server Implementation ----

# Initialize database
logger.info("Starting Medical Knowledge MCP Server initialization...")
try:
    medical_db = MedicalDatabase(CHROMA_DB_PATH)
except Exception as e:
    logger.error(f"FATAL: Cannot start server - database initialization failed")
    sys.exit(1)

# Create MCP server
server = Server("medical-knowledge-server")
logger.info("MCP Server instance created")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="search_medical_knowledge",
            description="Search the medical knowledge database for information about diseases, conditions, treatments, and medical topics. Returns relevant medical information with sources and relevance scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The medical question or topic to search for"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of results to return (default: 5, max: 10)",
                        "default": 5
                    },
                    "max_length": {
                        "type": "number",
                        "description": "Maximum length of content snippets in characters (default: 500, max: 2000)",
                        "default": 500
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_by_condition",
            description="Search specifically for information about a medical condition, including symptoms, diagnosis, and treatment options.",
            inputSchema={
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "The name of the medical condition (e.g., 'diabetes', 'hypertension', 'asthma')"
                    },
                    "max_length": {
                        "type": "number",
                        "description": "Maximum length of content snippets in characters (default: 400, max: 2000)",
                        "default": 400
                    }
                },
                "required": ["condition"]
            }
        ),
        Tool(
            name="search_by_treatment",
            description="Search for information about medical treatments, therapies, procedures, or medications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "treatment": {
                        "type": "string",
                        "description": "The name of the treatment, therapy, or medication"
                    },
                    "max_length": {
                        "type": "number",
                        "description": "Maximum length of content snippets in characters (default: 400, max: 2000)",
                        "default": 400
                    }
                },
                "required": ["treatment"]
            }
        ),
        Tool(
            name="search_by_symptom",
            description="Search for information about symptoms and their potential causes or related conditions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symptom": {
                        "type": "string",
                        "description": "The symptom to search for (e.g., 'fever', 'chest pain', 'headache')"
                    },
                    "max_length": {
                        "type": "number",
                        "description": "Maximum length of content snippets in characters (default: 400, max: 2000)",
                        "default": 400
                    }
                },
                "required": ["symptom"]
            }
        ),
        Tool(
            name="get_database_info",
            description="Get information about the medical knowledge database, including statistics and status.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from MCP clients."""
    logger.info("=" * 60)
    logger.info(f"TOOL CALLED: {name}")
    logger.info(f"Arguments: {arguments}")
    logger.info("=" * 60)
    
    try:
        if name == "search_medical_knowledge":
            query = arguments.get("query")
            top_k = min(arguments.get("top_k", 5), 10)  # Cap at 10
            max_length = min(arguments.get("max_length", 500), 2000)  # Cap at 2000
            
            if not query:
                return [TextContent(
                    type="text",
                    text="Error: 'query' parameter is required"
                )]
            
            result = medical_db.search_knowledge(query, top_k)
            
            # Format response
            if result["results_count"] == 0:
                response = f"No results found for query: '{query}'"
                if "error" in result:
                    response += f"\nError: {result['error']}"
            else:
                response = f"Medical Knowledge Search Results for: '{query}'\n"
                response += f"Found {result['results_count']} relevant documents\n\n"
                
                for r in result["results"]:
                    response += f"--- Result {r['rank']} (Relevance: {r['relevance_score']}) ---\n"
                    response += f"Source: {r['source']}\n"
                    content = r['content']
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    response += f"{content}\n\n"
            
            logger.info(f"✓ Response generated ({len(response)} chars)")
            return [TextContent(type="text", text=response)]
        
        elif name == "search_by_condition":
            condition = arguments.get("condition")
            max_length = min(arguments.get("max_length", 400), 2000)  # Cap at 2000
            
            if not condition:
                return [TextContent(
                    type="text",
                    text="Error: 'condition' parameter is required"
                )]
            
            result = medical_db.search_by_condition(condition)
            
            response = f"Medical Condition Search: '{condition}'\n\n"
            
            if result["results_count"] == 0:
                response += "No information found for this condition."
            else:
                for r in result["results"]:
                    response += f"--- {r['source']} (Relevance: {r['relevance_score']}) ---\n"
                    content = r['content']
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    response += f"{content}\n\n"
            
            logger.info(f"✓ Condition search complete")
            return [TextContent(type="text", text=response)]
        
        elif name == "search_by_treatment":
            treatment = arguments.get("treatment")
            max_length = min(arguments.get("max_length", 400), 2000)  # Cap at 2000
            
            if not treatment:
                return [TextContent(
                    type="text",
                    text="Error: 'treatment' parameter is required"
                )]
            
            result = medical_db.search_by_treatment(treatment)
            
            response = f"Treatment Search: '{treatment}'\n\n"
            
            if result["results_count"] == 0:
                response += "No information found for this treatment."
            else:
                for r in result["results"]:
                    response += f"--- {r['source']} (Relevance: {r['relevance_score']}) ---\n"
                    content = r['content']
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    response += f"{content}\n\n"
            
            logger.info(f"✓ Treatment search complete")
            return [TextContent(type="text", text=response)]
        
        elif name == "search_by_symptom":
            symptom = arguments.get("symptom")
            max_length = min(arguments.get("max_length", 400), 2000)  # Cap at 2000
            
            if not symptom:
                return [TextContent(
                    type="text",
                    text="Error: 'symptom' parameter is required"
                )]
            
            result = medical_db.search_by_symptom(symptom)
            
            response = f"Symptom Search: '{symptom}'\n\n"
            
            if result["results_count"] == 0:
                response += "No information found for this symptom."
            else:
                for r in result["results"]:
                    response += f"--- {r['source']} (Relevance: {r['relevance_score']}) ---\n"
                    content = r['content']
                    if len(content) > max_length:
                        content = content[:max_length] + "..."
                    response += f"{content}\n\n"
            
            logger.info(f"✓ Symptom search complete")
            return [TextContent(type="text", text=response)]
        
        elif name == "get_database_info":
            stats = medical_db.get_statistics()
            
            response = "Medical Knowledge Database Information\n\n"
            response += f"Status: {stats.get('status', 'unknown')}\n"
            response += f"Total Documents: {stats.get('total_documents', 0):,}\n"
            response += f"Collection: {stats.get('collection_name', 'N/A')}\n"
            response += f"Embedding Model: {stats.get('embedding_model', 'N/A')}\n"
            response += f"Database Path: {stats.get('database_path', 'N/A')}\n"
            response += f"Cache Size: {stats.get('cache_size', 0)}\n"
            
            if "error" in stats:
                response += f"\nError: {stats['error']}"
            
            logger.info(f"✓ Database info retrieved")
            return [TextContent(type="text", text=response)]
        
        else:
            return [TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}'"
            )]
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        logger.error(f"Tool: {name}")
        logger.error(f"Arguments: {arguments}")
        return [TextContent(
            type="text",
            text=f"Error executing tool '{name}': {str(e)}"
        )]

# Add HTTP server for client communication
from flask import Flask, request, jsonify

# Create HTTP server
http_app = Flask(__name__)

@http_app.route('/mcp_tool', methods=['POST'])
def mcp_tool_endpoint():
    """HTTP endpoint for MCP tool calls"""
    try:
        data = request.get_json()
        tool_name = data.get('tool_name')
        arguments = data.get('arguments', {})
        
        logger.info(f"HTTP MCP Tool Call: {tool_name} with args: {arguments}")
        
        # Call the appropriate method
        if tool_name == "search_medical_knowledge":
            query = arguments.get('query', '')
            top_k = arguments.get('top_k', 5)
            result = medical_db.search_knowledge(query, top_k)
        elif tool_name == "search_by_condition":
            condition = arguments.get('condition', '')
            result = medical_db.search_by_condition(condition)
        elif tool_name == "search_by_treatment":
            treatment = arguments.get('treatment', '')
            result = medical_db.search_by_treatment(treatment)
        elif tool_name == "search_by_symptom":
            symptom = arguments.get('symptom', '')
            result = medical_db.search_by_symptom(symptom)
        elif tool_name == "get_database_info":
            result = medical_db.get_statistics()
        else:
            result = {"error": "Unknown tool"}
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"HTTP MCP Tool Error: {e}")
        return jsonify({"error": str(e)}), 500

@http_app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "database": "connected",
        "tools": ["search_medical_knowledge", "search_by_condition", "search_by_treatment", "search_by_symptom", "get_database_info"]
    })

def run_http_server():
    """Run the HTTP server in a separate thread"""
    logger.info("🌐 Starting HTTP server on port 8000...")
    http_app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

async def main():
    """Main entry point for MCP server."""
    logger.info("=" * 60)
    logger.info("Medical Knowledge MCP Server Ready!")
    logger.info("=" * 60)
    logger.info("Available tools:")
    logger.info("  1. search_medical_knowledge - General medical search")
    logger.info("  2. search_by_condition - Search by medical condition")
    logger.info("  3. search_by_treatment - Search by treatment")
    logger.info("  4. search_by_symptom - Search by symptom")
    logger.info("  5. get_database_info - Database statistics")
    logger.info("=" * 60)
    logger.info("Starting HTTP server for client communication...")
    logger.info("HTTP API: http://localhost:8000/mcp_tool")
    logger.info("=" * 60)
    
    # Start HTTP server in a separate thread
    import threading
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    logger.info("Waiting for MCP client connection via stdio...")
    logger.info("(This server uses stdin/stdout for MCP communication)")
    logger.info("=" * 60)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)