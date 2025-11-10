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
            
            # Enhanced web search triggers
            should_use_web_search = False
            
            # Check for no results
            if "No results found" in result or "No relevant medical information found" in result or "Found 0 results" in result:
                logger.info("🌐 No database results - triggering web search")
                should_use_web_search = True
            
            # Check for insufficient data (very short response)
            elif len(result) < 200:
                logger.info("🌐 Insufficient database data - triggering web search")
                should_use_web_search = True
            
            # Check for low result count
            elif "Found 1 results" in result or "Found 2 results" in result:
                logger.info("🌐 Low result count - adding web search")
                should_use_web_search = True
            
            # Check for specific conditions that often lack data
            query_lower = query.lower()
            if any(term in query_lower for term in ['hip labral', 'fai', 'bursitis', 'teres major', 'uncommon', 'rare']):
                logger.info("🌐 Special condition detected - adding web search")
                should_use_web_search = True
            
            if should_use_web_search:
                web_result = self._web_search_medical_info(query)
                if web_result:
                    return f"Database Results:\n{result}\n\nWeb Search Results:\n{web_result}"
            
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
                input=f"Search for current physiotherapy and rehabilitation information about: {query}. Focus on musculoskeletal conditions, therapeutic exercises, rehabilitation protocols, movement analysis, and evidence-based physiotherapy research from reliable medical sources."
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
                
                # Look for specific body parts and conditions (physiotherapy-focused)
                body_parts = ['adductor', 'hip', 'groin', 'thigh', 'leg', 'knee', 'ankle', 'foot', 'back', 'spine', 'shoulder', 'arm', 'wrist', 'elbow', 'neck', 'cervical', 'lumbar', 'thoracic', 'pelvis', 'sacrum', 'patella', 'tibia', 'fibula', 'humerus', 'radius', 'ulna', 'scapula', 'clavicle', 'sternum', 'ribs']
                conditions = ['strain', 'sprain', 'injury', 'pain', 'tear', 'tendinitis', 'bursitis', 'arthritis', 'tendinopathy', 'tendinosis', 'myofascial', 'trigger point', 'muscle imbalance', 'postural dysfunction', 'movement dysfunction', 'instability', 'impingement', 'subluxation', 'dislocation', 'fracture', 'contusion', 'hematoma', 'edema', 'inflammation', 'stiffness', 'weakness', 'atrophy', 'hypertrophy']
                
                # Extract specific medical terms
                for body_part in body_parts:
                    if body_part in content:
                        specific_conditions.append(body_part)
                
                for condition in conditions:
                    if condition in content:
                        specific_conditions.append(condition)
                
                # If message contains medical terms, include it (physiotherapy-focused)
                physio_terms = ['pain', 'injury', 'condition', 'treatment', 'exercise', 'symptom', 'strain', 'sprain', 'rehabilitation', 'therapy', 'physiotherapy', 'physical therapy', 'movement', 'posture', 'gait', 'range of motion', 'strength', 'flexibility', 'mobility', 'stability', 'balance', 'coordination', 'proprioception', 'manual therapy', 'therapeutic exercise', 'assessment', 'evaluation', 'intervention', 'protocol', 'progression', 'regression']
                if any(term in content for term in physio_terms):
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
            model="gpt-4",
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
                description="Search medical database via MCP server for physiotherapy information, therapeutic exercises, rehabilitation protocols, movement analysis, and musculoskeletal conditions. Input: physiotherapy question, exercise query, or rehabilitation topic.",
                func=lambda query: self.mcp_client.search_medical_knowledge(query)
            ),
            Tool(
                name="search_condition",
                description="Search for musculoskeletal conditions and injuries via MCP server. Input: condition name (e.g., 'adductor strain', 'rotator cuff tear', 'patellofemoral pain syndrome').",
                func=lambda condition: self.mcp_client.search_by_condition(condition)
            ),
            Tool(
                name="search_treatment",
                description="Search for physiotherapy treatments and rehabilitation interventions via MCP server. Input: treatment name (e.g., 'manual therapy', 'therapeutic exercise', 'electrotherapy', 'gait training').",
                func=lambda treatment: self.mcp_client.search_by_treatment(treatment)
            ),
            Tool(
                name="search_symptom",
                description="Search for musculoskeletal symptoms and movement dysfunctions via MCP server. Input: symptom name (e.g., 'pain', 'stiffness', 'weakness', 'instability', 'limited range of motion').",
                func=lambda symptom: self.mcp_client.search_by_symptom(symptom)
            ),
            Tool(
                name="web_search_medical",
                description="Search the web for current physiotherapy research, new treatment protocols, or recent rehabilitation studies when database has no results. Use for evidence-based practice updates or when medical database search fails. Input: physiotherapy question or research topic.",
                func=lambda query: self.mcp_client._web_search_medical_info(query)
            )
        ]
    
    def _setup_chains(self):
        """Setup RetrievalQA with MCP retriever."""
        qa_prompt = PromptTemplate(
            template="""Physiotherapy Context: {context}

Question: {question}

Provide a concise, accurate physiotherapy-focused answer based on the context. Use proper physiotherapy terminology and focus on musculoskeletal conditions, rehabilitation, and therapeutic interventions. Be brief but informative with evidence-based recommendations.

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
        """Setup agent with MCP tools - MEMORY DISABLED."""
        # MEMORY DISABLED - Each query is treated independently
        # This prevents the agent from using previous conversation context
        # which was causing incorrect diagnoses
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=100,  # Minimal memory (effectively disabled)
            memory_key_prefix="",
            input_key="input",
            output_key="output"
        )
        
        # Create condensed system message focused on physiotherapy
        # MEMORY DISABLED - Each query is independent
        system_message = """You are a physiotherapy-focused medical assistant. Use MCP server for medical data with web search fallback.

PHYSIOTHERAPY FOCUS: You specialize in musculoskeletal conditions, rehabilitation, exercise therapy, and movement analysis. Use physiotherapy terminology and approaches.

TOOLS: search_medical_knowledge, search_condition, search_treatment, search_symptom, web_search_medical

CRITICAL FORMAT - You MUST use this exact format (no parentheses, no quotes):
Thought: [your reasoning here]
Action: search_condition
Action Input: knee pain and swelling
Observation: [tool result will appear here]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: [final reasoning]
Final Answer: [your complete answer]

WRONG FORMAT (DO NOT USE):
Action: search_condition('knee pain and swelling')  ❌ WRONG
Action: search_condition("knee pain")  ❌ WRONG

CORRECT FORMAT:
Action: search_condition
Action Input: knee pain and swelling  ✓ CORRECT

IMPORTANT: Treat EACH query as INDEPENDENT. Do NOT use previous conversation context. Base your diagnosis ONLY on the current symptoms mentioned.

PHYSIOTHERAPY TERMINOLOGY: Use terms like:
- Musculoskeletal conditions (strains, sprains, tendinopathies, bursitis)
- Movement patterns (gait analysis, posture assessment, range of motion)
- Rehabilitation phases (acute, subacute, chronic, return to sport)
- Exercise classifications (strengthening, stretching, proprioceptive, functional)
- Assessment techniques (palpation, special tests, functional movement screening)
- Treatment modalities (manual therapy, therapeutic exercise, electrotherapy)

WEB SEARCH TRIGGERS (CRITICAL - Always use web_search_medical when):
- Medical database returns "No results found" or "Found 0 results"
- Results have very low relevance scores (< 50%)
- Dealing with hip conditions (labral tears, FAI, impingement)
- Dealing with rare conditions (bursitis, uncommon injuries)
- First search yields insufficient information (< 3 quality results)
- After 1 unsuccessful database search, IMMEDIATELY try web_search_medical

IF NO DATA FOUND: You MUST still provide a Final Answer with 5 differential diagnoses based on the clinical presentation in the query. Use your medical knowledge and the symptoms described. NEVER say "search did not provide" - always generate the 5-point diagnosis list.

CRITICAL - DIAGNOSIS FORMAT (YOU MUST FOLLOW THIS EXACTLY):

FIRST: Check if this is POST-OPERATIVE (had surgery):
- Look for: "post op", "post surgical", "ACLR", "reconstruction", "surgery", "weeks/months post"
- If YES → Write 1 status statement + 4 complications/concerns (if symptoms present)
- If NO → Write 5 differential diagnoses

POST-OPERATIVE FORMAT (if patient had surgery):
✓ "R knee ACL arthroscopic reconstruction - Post op status 8 weeks"
✓ "Left ACLR post surgical status 5 years (biomechanical deficits)"
✓ "Post surgical R knee instability"

PRE-OPERATIVE FORMAT (no surgery yet):
✓ "Suspected R knee medial meniscus tear?"
✓ "Suspected L hip labral injury + FAI?"
✓ "Suspected B/L knee OA (patellofemoral)?"
✓ "Suspected R shoulder rotator cuff tendinopathy?"
✓ "Suspected L IT band syndrome secondary to biomechanical deficit?"

❌ WRONG EXAMPLES (Too Verbose):
✗ "Suspected Left femoral hypomobility syndrome with superior glide?"
✗ "Suspected Left LBP and posterior thigh pain?"
✗ "Suspected lumbar spine or soft tissue involvement in left hip pain?"

KEY RULES:
- Be EXTREMELY concise - use specific condition NAMES, not descriptions
- Use abbreviations: R, L, B/L, ACL, ACLR, FAI, OA, PFPS, IT band
- Format for suspected: "Suspected [Body Part] [Condition]?"
- Format for post-op: "[Body Part] [Surgery Type] - Post op status [timeframe]"
- Add "?" for uncertainty (except confirmed post-op status)
- Use "+" to combine conditions
- Can add brief context like "secondary to" if relevant
- NO verbose descriptions - just condition names or surgical status

NEW REQUIREMENT - PROBABILITY SCORES:
After listing each diagnosis, add a probability/relevance score (0-100%) in parentheses to indicate how likely each diagnosis is based on the clinical presentation.

FORMAT WITH PROBABILITY:
1. Suspected R knee medial meniscus tear? (85%)
2. Suspected R knee cartilage irritation? (70%)
3. Suspected L knee PFPS? (60%)
4. Suspected bilateral patellofemoral dysfunction? (45%)
5. Suspected IT band syndrome bilateral? (30%)

REQUIRED FIELDS SECTION (ONLY FOR PATIENT DIAGNOSIS):
ONLY add this section if you are diagnosing a SPECIFIC PATIENT with actual clinical data (chief complaints, test results, assessment notes).

DO NOT add this section if:
- User is asking a general medical question ("What is tennis elbow?")
- User is asking about a condition in general ("Tell me about ACL tears")
- No patient data is provided in the query

IF diagnosing a patient, list ONLY the SPECIFIC JSON FIELD PATHS that you ACTUALLY USED:

REQUIRED FIELDS FOR DIAGNOSIS:
[List only the field paths you actually used]

EXAMPLE:
- records.clinicalDetails.chiefComplaints
- records.objectiveAssessment.tests[].testName (Clark's Test, Thessaly's Test)
- records.subjectiveAssessment.assessment

RESPONSE: Extract specific info from search results. If user asks "10 exercises", provide exactly 10 numbered items with proper physiotherapy progression.

PROCESS: 1) Review query 2) Search database 3) If poor results, use web search 4) Extract info 5) ALWAYS provide numbered diagnosis list with probability scores 6) Add required fields section 7) Format properly."""
        
        self.agent = initialize_agent(
            tools=self.mcp_tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            max_execution_time=45,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            agent_kwargs={"prefix": system_message}
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
        # CLEAR MEMORY before each query to ensure independence
        medical_system.clear_memory()
        logger.info("🧹 Memory cleared - treating query independently")
        
        # Use question directly without context enhancement
        # This ensures each query is treated independently
        enhanced_question = question
        
        logger.info(f"🔄 Invoking LangChain agent with MCP tools (NO MEMORY)...")
        result = medical_system.agent.invoke({"input": enhanced_question})
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Agent completed in {processing_time:.2f}s")
        logger.info(f"📊 Response length: {len(result['output'])} characters")
        
        return jsonify({
            "question": question,
            "answer": result["output"],
            "approach": "MCP-Integrated LangChain Agent (NO MEMORY)",
            "processing_time": f"{processing_time:.2f}s",
            "optimizations": ["MCP Client", "No Code Duplication", "Centralized Database", "Independent Queries"],
            "memory_status": "DISABLED for accurate diagnoses"
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
        "title": "MCP-Integrated Physiotherapy API with Web Search",
        "description": "Flask server that connects to mcp_medical_server.py via MCP protocol with OpenAI web search fallback, specialized for physiotherapy and rehabilitation",
        "architecture": {
            "flask_server": "HTTP REST API endpoints",
            "mcp_client": "Connects to mcp_medical_server.py",
            "database": "Shared ChromaDB via MCP server",
            "web_search": "OpenAI web search for physiotherapy information when database has no results"
        },
        "endpoints": {
            "POST /fast_ask": "Fast MCP server search with web search fallback",
            "POST /agent_ask": "LangChain agent with MCP tools and web search (physiotherapy-focused)",
            "POST /retrieval_ask": "RetrievalQA with MCP retriever"
        },
        "features": [
            "Physiotherapy-focused medical database search via MCP server",
            "Web search fallback for current rehabilitation research",
            "Context-aware conversation memory with physiotherapy terminology",
            "Multiple search strategies (musculoskeletal conditions, treatments, symptoms)",
            "Real-time physiotherapy information from web",
            "Specialized in movement analysis and therapeutic exercise"
        ],
        "benefits": [
            "No code duplication",
            "Centralized database access",
            "MCP protocol communication",
            "Independent server scaling",
            "Web search for current physiotherapy research",
            "Comprehensive musculoskeletal knowledge coverage",
            "Physiotherapy-specific terminology and approaches"
        ]
    })

if __name__ == '__main__':
    logger.info("🏥 MCP-INTEGRATED Physiotherapy API Server with Web Search")
    logger.info("=" * 60)
    logger.info("🔗 Architecture:")
    logger.info("  - Flask Server (HTTP API)")
    logger.info("  - MCP Client (connects to mcp_medical_server.py)")
    logger.info("  - Shared ChromaDB (via MCP server)")
    logger.info("  - OpenAI Web Search (fallback for physiotherapy research)")
    logger.info("=" * 60)
    logger.info("🚀 Server: http://localhost:5000")
    logger.info("⚡ Fast Search: POST /fast_ask (with web search fallback)")
    logger.info("🤖 Smart Agent: POST /agent_ask (physiotherapy-focused with web search)")
    logger.info("🔍 RetrievalQA: POST /retrieval_ask")
    logger.info("🌐 Web Search: Automatic when database has no results")
    logger.info("🏃 Physiotherapy Focus: Musculoskeletal conditions, rehabilitation, therapeutic exercise")
    logger.info("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
