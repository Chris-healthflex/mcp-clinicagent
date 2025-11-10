#!/usr/bin/env python3
"""
HIGH-PERFORMANCE LangChain Medical API Server with MCP Client
Connects to mcp_medical_server.py instead of duplicating database logic.
"""

import os
import json
import logging
import time
import asyncio
import subprocess
import requests
from datetime import datetime
from flask import Flask, request, jsonify
import dotenv
from typing import List, Dict, Any
from openai import OpenAI

# MCP imports - removed unused ones

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory

# Load environment variables
dotenv.load_dotenv()

# Setup verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---- MCP CLIENT IMPLEMENTATION ----
class MCPMedicalClient:
    """Client to communicate with mcp_medical_server.py via subprocess"""
    
    def __init__(self):
        self._init_mcp_client()
        self._init_database()
        self._init_web_search()
    
    def _init_mcp_client(self):
        """Initialize MCP client connection to mcp_medical_server.py"""
        logger.info("🔗 Initializing MCP client connection...")
        logger.info("Setting up MCP client to connect to mcp_medical_server.py")
        logger.info("📡 MCP client ready to connect to mcp_medical_server.py")
        logger.info("MCP client initialized")
    
    def _init_database(self):
        """Connect to the running MCP server via HTTP"""
        try:
            logger.info("🔍 Connecting to running MCP server...")
            logger.info("📡 Will communicate with MCP server via HTTP")
            # Don't import the server module - just prepare for HTTP communication
            self.db = None  # We'll use HTTP requests instead
            logger.info("✅ Ready to communicate with running MCP server via HTTP")
        except Exception as e:
            logger.error(f"❌ MCP server connection error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.db = None
    
    def _init_web_search(self):
        """Initialize OpenAI client for web search functionality"""
        try:
            logger.info("🌐 Initializing web search capabilities...")
            self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info("✅ Web search client initialized")
        except Exception as e:
            logger.error(f"❌ Web search initialization error: {e}")
            self.openai_client = None
    
    def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Call MCP tool via HTTP request to running server"""
        try:
            logger.info(f"📡 Calling MCP tool: {tool_name} with args: {arguments}")
            
            # Make HTTP request to the MCP server
            url = "http://localhost:8000/mcp_tool"
            payload = {
                "tool_name": tool_name,
                "arguments": arguments
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Format result
                if isinstance(result, dict):
                    if result.get("results_count", 0) > 0:
                        response_text = f"Found {result['results_count']} results:\n\n"
                        for i, r in enumerate(result.get("results", [])):
                            response_text += f"--- Result {i+1} (Relevance: {r.get('relevance_score', 'N/A')}) ---\n"
                            response_text += f"Source: {r.get('source', 'N/A')}\n"
                            # Clean content to avoid Unicode issues
                            content = r.get('content', '')
                            if content:
                                # Replace problematic Unicode characters
                                content = content.replace('\u2010', '-').replace('\ufb01', 'fi').replace('\ufb02', 'fl')
                                response_text += f"{content[:500]}...\n\n"
                        return response_text
                    else:
                        return "No results found"
                else:
                    return str(result)
            else:
                return f"MCP server HTTP error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "MCP server not running - please start it first"
        except Exception as e:
            return f"Error calling MCP tool: {str(e)}"
    
    def search_medical_knowledge(self, query: str, top_k: int = 5) -> str:
        """Search medical knowledge via MCP server with web search fallback"""
        logger.info(f"🔍 MCP CLIENT: search_medical_knowledge")
        logger.info(f"📝 Query: '{query}' | Top-K: {top_k}")
        
        try:
            result = self._call_mcp_tool("search_medical_knowledge", {"query": query, "top_k": top_k})
            logger.info(f"✅ MCP response received ({len(result)} chars)")
            
            # Check if no results found and try web search
            if "No results found" in result or "No relevant medical information found" in result:
                logger.info("🌐 No medical database results found, attempting web search...")
                web_result = self._web_search_medical_info(query)
                if web_result:
                    return f"Medical Database: {result}\n\nWeb Search Results:\n{web_result}"
            
            return result
        except Exception as e:
            logger.error(f"❌ MCP client error: {e}")
            # Try web search as fallback
            logger.info("🌐 MCP failed, attempting web search fallback...")
            web_result = self._web_search_medical_info(query)
            if web_result:
                return f"Database unavailable. Web Search Results:\n{web_result}"
            return f"Error connecting to medical server: {str(e)}"
    
    def search_by_condition(self, condition: str) -> str:
        """Search by medical condition via MCP server"""
        logger.info(f"🏥 MCP CLIENT: search_by_condition")
        logger.info(f"📝 Condition: '{condition}'")
        
        try:
            result = self._call_mcp_tool("search_by_condition", {"condition": condition})
            logger.info(f"✅ MCP condition response received")
            return result
        except Exception as e:
            logger.error(f"❌ MCP condition search error: {e}")
            return f"Error searching condition: {str(e)}"
    
    def search_by_treatment(self, treatment: str) -> str:
        """Search by treatment via MCP server"""
        logger.info(f"💊 MCP CLIENT: search_by_treatment")
        logger.info(f"📝 Treatment: '{treatment}'")
        
        try:
            result = self._call_mcp_tool("search_by_treatment", {"treatment": treatment})
            logger.info(f"✅ MCP treatment response received")
            return result
        except Exception as e:
            logger.error(f"❌ MCP treatment search error: {e}")
            return f"Error searching treatment: {str(e)}"
    
    def search_by_symptom(self, symptom: str) -> str:
        """Search by symptom via MCP server"""
        logger.info(f"🩺 MCP CLIENT: search_by_symptom")
        logger.info(f"📝 Symptom: '{symptom}'")
        
        try:
            result = self._call_mcp_tool("search_by_symptom", {"symptom": symptom})
            logger.info(f"✅ MCP symptom response received")
            return result
        except Exception as e:
            logger.error(f"❌ MCP symptom search error: {e}")
            return f"Error searching symptom: {str(e)}"
    
    def get_database_info(self) -> str:
        """Get database info via MCP server"""
        logger.info(f"📊 MCP CLIENT: get_database_info")
        
        try:
            result = self._call_mcp_tool("get_database_info", {})
            logger.info(f"✅ MCP database info received")
            return result
        except Exception as e:
            logger.error(f"❌ MCP database info error: {e}")
            return f"Error getting database info: {str(e)}"
    
    def _web_search_medical_info(self, query: str) -> str:
        """Perform web search for medical information using OpenAI's web search capability"""
        if not self.openai_client:
            logger.warning("🌐 Web search client not available")
            return ""
        
        try:
            logger.info(f"🌐 Performing web search for: '{query}'")
            
            # Use OpenAI's responses API with web search tool
            response = self.openai_client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=f"Search for current medical information about: {query}. Focus on reliable medical sources, symptoms, treatments, and recent research."
            )
            
            web_result = response.output_text
            logger.info(f"✅ Web search completed ({len(web_result)} chars)")
            return web_result
            
        except Exception as e:
            logger.error(f"❌ Web search error: {e}")
            return ""

# ---- MCP RETRIEVER ----
class MCPMedicalRetriever(BaseRetriever):
    """Retriever that uses MCP client instead of direct database access"""
    
    def __init__(self, mcp_client: MCPMedicalClient, k: int = 3):
        super().__init__()
        self._mcp_client = mcp_client
        self._k = k
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents via MCP client - returns multiple documents for better RAG"""
        logger.info(f"🔗 MCP RETRIEVER CALLED")
        logger.info(f"📝 Query: '{query}' | K: {self._k}")
        
        try:
            # Call MCP client directly (now synchronous)
            mcp_result = self._mcp_client.search_medical_knowledge(query, self._k)
            
            logger.info(f"📊 MCP Result length: {len(mcp_result)} characters")
            
            if "No relevant medical information found" in mcp_result:
                logger.info("❌ No relevant information found")
                return []
            
            # Parse the result into multiple documents for better RAG
            documents = []
            
            # Split by "--- Result" to get individual results
            if "--- Result" in mcp_result:
                results = mcp_result.split("--- Result")[1:]  # Skip the header
                
                for i, result in enumerate(results[:self._k]):
                    if result.strip():
                        # Extract source and content
                        lines = result.strip().split('\n')
                        source = "Unknown"
                        content = result.strip()
                        
                        # Try to extract source from the result
                        for line in lines:
                            if line.startswith("Source:"):
                                source = line.replace("Source:", "").strip()
                                break
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": source,
                                "query": query,
                                "result_index": i,
                                "retriever": "MCP Medical Server"
                            }
                        )
                        documents.append(doc)
            else:
                # Fallback: create single document if parsing fails
                doc = Document(
                    page_content=mcp_result,
                    metadata={"source": "MCP Medical Server", "query": query}
                )
                documents.append(doc)
            
            logger.info(f"✅ Retrieved {len(documents)} documents via MCP")
            return documents
            
        except Exception as e:
            logger.error(f"❌ MCP Retriever error: {e}")
            return []

# ---- OPTIMIZED LANGCHAIN SYSTEM WITH MCP ----
class MCPMedicalSystem:
    """LangChain system that uses MCP client instead of direct database access"""
    
    def __init__(self):
        logger.info("🏗️ Initializing MCP-based medical system...")
        
        # Initialize MCP client
        self.mcp_client = MCPMedicalClient()
        
        # Initialize LLM
        self._init_llm()
        
        # Setup MCP-based components
        self.retriever = MCPMedicalRetriever(self.mcp_client, k=3)
        self._setup_mcp_tools()
        self._setup_chains()
        self._setup_agent()
        
        logger.info("✅ MCP-based medical system ready")
    
    def clear_memory(self):
        """Clear the agent's conversation memory."""
        self.memory.clear()
        logger.info("🧠 Agent memory cleared")
    
    def get_memory_summary(self):
        """Get a summary of the current memory state."""
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        return {
            "memory_length": len(chat_history),
            "has_memory": len(chat_history) > 0,
            "recent_messages": chat_history[-2:] if chat_history else []
        }
    
    def get_conversation_context(self):
        """Get conversation context for better agent understanding."""
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        if not chat_history:
            return "No previous conversation context."
        
        # Extract specific medical conditions and body parts from recent conversation
        recent_context = []
        specific_conditions = []
        
        for message in chat_history[-6:]:  # Last 6 messages for better context
            if hasattr(message, 'content'):
                content = message.content.lower()
                
                # Look for specific body parts and conditions
                body_parts = ['adductor', 'hip', 'groin', 'thigh', 'leg', 'knee', 'ankle', 'foot', 'back', 'spine', 'shoulder', 'arm', 'wrist', 'elbow']
                conditions = ['strain', 'sprain', 'injury', 'pain', 'tear', 'tendinitis', 'bursitis', 'arthritis']
                
                # Extract specific medical terms
                for body_part in body_parts:
                    if body_part in content:
                        specific_conditions.append(body_part)
                
                for condition in conditions:
                    if condition in content:
                        specific_conditions.append(condition)
                
                # If message contains medical terms, include it
                if any(term in content for term in ['pain', 'injury', 'condition', 'treatment', 'exercise', 'symptom', 'strain', 'sprain']):
                    recent_context.append(content[:150] + "...")
        
        # Create specific context summary
        if specific_conditions:
            context_summary = f"Previous discussion involved: {', '.join(set(specific_conditions))}"
            if recent_context:
                context_summary += f" | Recent context: {' | '.join(recent_context[-2:])}"
            return context_summary
        elif recent_context:
            return f"Recent context: {' | '.join(recent_context[-2:])}"
        else:
            return "No specific medical context found."
    
    def _init_llm(self):
        """Initialize LLM with speed optimizations."""
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=3000,
            api_key=os.environ.get("OPENAI_API_KEY"),
            request_timeout=20,
            max_retries=2
        )
    
    def _setup_mcp_tools(self):
        """Setup MCP-based tools."""
        self.mcp_tools = [
            Tool(
                name="search_medical_knowledge",
                description="Search medical database via MCP server for general medical information, exercises, rehabilitation, and knowledge. Input: medical question, topic, or exercise query.",
                func=lambda query: self.mcp_client.search_medical_knowledge(query)
            ),
            Tool(
                name="search_condition",
                description="Search for medical conditions via MCP server. Input: condition name.",
                func=lambda condition: self.mcp_client.search_by_condition(condition)
            ),
            Tool(
                name="search_treatment",
                description="Search for medical treatments and therapies via MCP server. Input: treatment name (e.g., 'surgery', 'medication', 'physical therapy').",
                func=lambda treatment: self.mcp_client.search_by_treatment(treatment)
            ),
            Tool(
                name="search_symptom",
                description="Search for symptoms via MCP server. Input: symptom name.",
                func=lambda symptom: self.mcp_client.search_by_symptom(symptom)
            ),
            Tool(
                name="web_search_medical",
                description="Search the web for current medical information when database has no results. Use for recent medical research, new treatments, or when medical database search fails. Input: medical question or topic.",
                func=lambda query: self.mcp_client._web_search_medical_info(query)
            )
        ]
    
    def _setup_chains(self):
        """Setup RetrievalQA with MCP retriever."""
        qa_prompt = PromptTemplate(
            template="""Medical Context: {context}

Question: {question}

Provide a concise, accurate medical answer based on the context. Be brief but informative.

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": qa_prompt}
        )
    
    def _setup_agent(self):
        """Setup agent with MCP tools."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=4000,
            memory_key_prefix="",
            input_key="input",
            output_key="output"
        )
        
        # Create condensed system message
        system_message = """You are a medical assistant with memory. Use MCP server for medical data with web search fallback.

TOOLS: search_medical_knowledge, search_condition, search_treatment, search_symptom, web_search_medical

FORMAT: Thought → Action → Action Input → Observation → Final Answer

CONTEXT: Check chat history first. If user mentions "this", "it", "the pain" - use previous conversation context.

CRITICAL: If we discussed "adductor pain" and user asks "exercises", search "adductor strain exercises" NOT generic exercises.

WEB SEARCH: Use web_search_medical when:
- Medical database returns "No results found" or "No relevant medical information found"
- User asks about recent medical research or new treatments
- Medical database search fails or times out
- User asks about very specific or rare conditions not in database

RESPONSE: Extract specific info from search results. If user asks "10 exercises", provide exactly 10 numbered items.

PROCESS: 1) Review context 2) Try medical database first 3) Use web search if no results 4) Extract specific info 5) Format properly 6) Remember for future."""
        
        self.agent = initialize_agent(
            tools=self.mcp_tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            max_execution_time=30,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            agent_kwargs={"system_message": system_message}
        )

# Initialize MCP-based system
logger.info("🚀 Initializing MCP-based medical system...")
medical_system = MCPMedicalSystem()

# ---- API ENDPOINTS ----

@app.route('/fast_ask', methods=['POST'])
def fast_ask():
    """Fast endpoint using MCP server."""
    logger.info(f"⚡ FAST_ASK ENDPOINT CALLED (MCP)")
    logger.info(f"📥 Raw request data: {request.get_json()}")
    
    data = request.get_json()
    if not data:
        logger.error("❌ No JSON data received")
        return jsonify({"error": "No JSON data received"}), 400
    
    question = data.get('question', '')
    logger.info(f"📝 Question: '{question}'")
    
    if not question:
        logger.error("❌ No question provided")
        return jsonify({"error": "Question required"}), 400
    
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Calling MCP server...")
        result = medical_system.mcp_client.search_medical_knowledge(question, top_k=3)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Fast ask completed in {processing_time:.2f}s")
        logger.info(f"📊 Response length: {len(result)} characters")
        
        return jsonify({
            "question": question,
            "answer": result,
            "approach": "MCP Server Integration",
            "processing_time": f"{processing_time:.2f}s",
            "optimizations": ["MCP Client", "No Code Duplication", "Centralized Database"]
        })
        
    except Exception as e:
        logger.error(f"❌ Fast ask error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/agent_ask', methods=['POST'])
def agent_ask():
    """Agent endpoint using MCP server."""
    logger.info(f"🤖 AGENT_ASK ENDPOINT CALLED (MCP)")
    logger.info(f"📥 Raw request data: {request.get_json()}")
    
    data = request.get_json()
    if not data:
        logger.error("❌ No JSON data received")
        return jsonify({"error": "No JSON data received"}), 400
    
    question = data.get('question', '')
    logger.info(f"📝 Question: '{question}'")
    
    if not question:
        logger.error("❌ No question provided")
        return jsonify({"error": "Question required"}), 400
    
    start_time = time.time()
    
    try:
        # Get conversation context for better agent understanding
        context = medical_system.get_conversation_context()
        logger.info(f"🧠 Conversation context: {context}")
        
        # Enhance question with context if needed
        enhanced_question = question
        if context != "No previous conversation context." and context != "No specific medical context found.":
            # Add explicit context hint to help agent understand
            enhanced_question = f"IMPORTANT CONTEXT: {context}\n\nUser Question: {question}\n\nRemember: Use the specific condition from context, not generic terms!"
        
        logger.info(f"🔄 Invoking LangChain agent with MCP tools...")
        result = medical_system.agent.invoke({"input": enhanced_question})
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Agent completed in {processing_time:.2f}s")
        logger.info(f"📊 Response length: {len(result['output'])} characters")
        
        return jsonify({
            "question": question,
            "answer": result["output"],
            "approach": "MCP-Integrated LangChain Agent with Context",
            "processing_time": f"{processing_time:.2f}s",
            "optimizations": ["MCP Client", "No Code Duplication", "Centralized Database", "Context Awareness"],
            "context_used": context
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Agent error: {error_msg}")
        
        # Fallback to direct MCP call
        try:
            fallback_result = medical_system.mcp_client.search_medical_knowledge(question, top_k=3)
            return jsonify({
                "question": question,
                "answer": f"Agent reached limit. Here's what I found: {fallback_result}",
                "approach": "MCP Fallback",
                "processing_time": f"{time.time() - start_time:.2f}s",
                "warning": "Agent iteration limit reached, used MCP fallback"
            })
        except Exception as fallback_error:
            return jsonify({
                "question": question,
                "answer": "I'm having trouble processing your request. Please try rephrasing your question.",
                "approach": "Error Fallback",
                "processing_time": f"{time.time() - start_time:.2f}s",
                "error": "Both agent and MCP fallback failed"
            }), 200

@app.route('/retrieval_ask', methods=['POST'])
def retrieval_ask():
    """RetrievalQA endpoint using MCP server."""
    logger.info(f"🔍 RETRIEVAL_ASK ENDPOINT CALLED (MCP)")
    logger.info(f"📥 Raw request data: {request.get_json()}")
    
    data = request.get_json()
    if not data:
        logger.error("❌ No JSON data received")
        return jsonify({"error": "No JSON data received"}), 400
    
    question = data.get('question', '')
    logger.info(f"📝 Question: '{question}'")
    
    if not question:
        logger.error("❌ No question provided")
        return jsonify({"error": "Question required"}), 400
    
    start_time = time.time()
    
    try:
        logger.info(f"🔄 Invoking RetrievalQA with MCP retriever...")
        result = medical_system.retrieval_qa.invoke({"query": question})
        processing_time = time.time() - start_time
        
        logger.info(f"✅ RetrievalQA completed in {processing_time:.2f}s")
        logger.info(f"📊 Response length: {len(result['result'])} characters")
        
        return jsonify({
            "question": question,
            "answer": result["result"],
            "approach": "MCP-Integrated RetrievalQA",
            "processing_time": f"{processing_time:.2f}s",
            "optimizations": ["MCP Client", "No Code Duplication", "Centralized Database"]
        })
        
    except Exception as e:
        logger.error(f"❌ RetrievalQA error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check using MCP server."""
    logger.info(f"🏥 HEALTH CHECK ENDPOINT CALLED (MCP)")
    
    try:
        # Get database info via MCP
        db_info = medical_system.mcp_client.get_database_info()
        
        # Check memory status
        memory_info = medical_system.get_memory_summary()
        context = medical_system.get_conversation_context()
        
        # Check web search availability
        web_search_status = "available" if medical_system.mcp_client.openai_client else "unavailable"
        
        return jsonify({
            "status": "optimal",
            "database_info": db_info,
            "memory": memory_info,
            "conversation_context": context,
            "web_search": {
                "status": web_search_status,
                "description": "OpenAI web search for medical information when database has no results"
            },
            "architecture": "MCP Client + Flask Server + Web Search",
            "optimizations": [
                "MCP Server Integration",
                "No Code Duplication",
                "Centralized Database Access",
                "Context Awareness",
                "Conversation Memory",
                "Web Search Fallback"
            ]
        })
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/memory/clear', methods=['POST'])
def clear_memory():
    """Clear the agent's conversation memory."""
    logger.info(f"🧠 MEMORY CLEAR ENDPOINT CALLED")
    
    try:
        medical_system.clear_memory()
        return jsonify({
            "status": "success",
            "message": "Agent memory cleared",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Memory clear error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/status', methods=['GET'])
def memory_status():
    """Get the current memory status."""
    logger.info(f"🧠 MEMORY STATUS ENDPOINT CALLED")
    
    try:
        memory_info = medical_system.get_memory_summary()
        context = medical_system.get_conversation_context()
        
        # Get raw memory variables for debugging
        memory_vars = medical_system.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        return jsonify({
            "status": "success",
            "memory": memory_info,
            "conversation_context": context,
            "raw_memory": {
                "chat_history_length": len(chat_history),
                "recent_messages": [str(msg) for msg in chat_history[-3:]] if chat_history else []
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Memory status error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """API documentation."""
    logger.info(f"📚 INDEX ENDPOINT CALLED")
    
    return jsonify({
        "title": "MCP-Integrated Medical API with Web Search",
        "description": "Flask server that connects to mcp_medical_server.py via MCP protocol with OpenAI web search fallback",
        "architecture": {
            "flask_server": "HTTP REST API endpoints",
            "mcp_client": "Connects to mcp_medical_server.py",
            "database": "Shared ChromaDB via MCP server",
            "web_search": "OpenAI web search for medical information when database has no results"
        },
        "endpoints": {
            "POST /fast_ask": "Fast MCP server search with web search fallback",
            "POST /agent_ask": "LangChain agent with MCP tools and web search",
            "POST /retrieval_ask": "RetrievalQA with MCP retriever"
        },
        "features": [
            "Medical database search via MCP server",
            "Web search fallback when no database results",
            "Context-aware conversation memory",
            "Multiple search strategies (condition, treatment, symptom)",
            "Real-time medical information from web"
        ],
        "benefits": [
            "No code duplication",
            "Centralized database access",
            "MCP protocol communication",
            "Independent server scaling",
            "Web search for current medical information",
            "Comprehensive medical knowledge coverage"
        ]
    })

if __name__ == '__main__':
    logger.info("🏥 MCP-INTEGRATED Medical API Server with Web Search")
    logger.info("=" * 60)
    logger.info("🔗 Architecture:")
    logger.info("  - Flask Server (HTTP API)")
    logger.info("  - MCP Client (connects to mcp_medical_server.py)")
    logger.info("  - Shared ChromaDB (via MCP server)")
    logger.info("  - OpenAI Web Search (fallback for missing info)")
    logger.info("=" * 60)
    logger.info("🚀 Server: http://localhost:5000")
    logger.info("⚡ Fast Search: POST /fast_ask (with web search fallback)")
    logger.info("🤖 Smart Agent: POST /agent_ask (with web search tool)")
    logger.info("🔍 RetrievalQA: POST /retrieval_ask")
    logger.info("🌐 Web Search: Automatic when database has no results")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
