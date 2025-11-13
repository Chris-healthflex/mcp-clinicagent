#!/usr/bin/env python3
"""
Provisional Diagnosis System using LangGraph + LangChain
Generates 5 provisional diagnoses with iterative MCP validation
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Setup logging first (before Flask imports)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Flask imports
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not available. Server mode will not work.")

from diagnosis_models import DiagnosisResult, FinalReport, ValidationQuery
from diagnosis_prompts import (
    get_query_builder_prompt,
    get_initial_diagnosis_prompt,
    get_validation_prompt,
    get_final_diagnosis_prompt
)
from mcp_stdio_client import MCPStdioClient
import requests

# Load environment variables
dotenv.load_dotenv()

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "o3")
MAX_ITERATIONS_PER_DIAGNOSIS = int(os.getenv("MAX_ITERATIONS_PER_DIAGNOSIS", "5"))
NUM_DIAGNOSES = int(os.getenv("NUM_DIAGNOSES", "5"))
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "./mcp_medical_server.py")
MCP_HTTP_URL = os.getenv("MCP_HTTP_URL", "http://localhost:8000")  # Use HTTP if server already running
OUTPUT_DIR = os.getenv("DIAGNOSIS_OUTPUT_DIR", "./provisional_users")
USE_MCP_HTTP = os.getenv("USE_MCP_HTTP", "true").lower() == "true"  # Use HTTP by default


class MCPHTTPClient:
    """HTTP-based MCP client for connecting to already-running MCP server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def search_medical_knowledge(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search medical knowledge base via HTTP."""
        try:
            response = requests.post(
                f"{self.base_url}/mcp_tool",
                json={
                    "tool_name": "search_medical_knowledge",
                    "arguments": {"query": query, "top_k": top_k}
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                # The HTTP endpoint returns the result directly (from medical_db.search_knowledge)
                # which has structure: {"query": ..., "results_count": ..., "results": [...]}
                return result
            else:
                return {"error": f"HTTP {response.status_code}", "results": [], "results_count": 0}
        except Exception as e:
            logger.error(f"HTTP MCP call error: {e}")
            return {"error": str(e), "results": [], "results_count": 0}
    
    async def search_by_condition(self, condition: str) -> Dict[str, Any]:
        """Search by condition via HTTP."""
        try:
            response = requests.post(
                f"{self.base_url}/mcp_tool",
                json={
                    "tool_name": "search_by_condition",
                    "arguments": {"condition": condition}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}", "results": []}
        except Exception as e:
            return {"error": str(e), "results": []}
    
    async def search_by_treatment(self, treatment: str) -> Dict[str, Any]:
        """Search by treatment via HTTP."""
        try:
            response = requests.post(
                f"{self.base_url}/mcp_tool",
                json={
                    "tool_name": "search_by_treatment",
                    "arguments": {"treatment": treatment}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}", "results": []}
        except Exception as e:
            return {"error": str(e), "results": []}
    
    async def search_by_symptom(self, symptom: str) -> Dict[str, Any]:
        """Search by symptom via HTTP."""
        try:
            response = requests.post(
                f"{self.base_url}/mcp_tool",
                json={
                    "tool_name": "search_by_symptom",
                    "arguments": {"symptom": symptom}
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}", "results": []}
        except Exception as e:
            return {"error": str(e), "results": []}
    
    async def disconnect(self):
        """No-op for HTTP client."""
        pass


class DiagnosisState(TypedDict):
    """State for the diagnosis graph."""
    patient_json: Dict[str, Any]
    patient_id: str
    combined_query: str
    initial_mcp_results: Dict[str, Any]
    current_diagnosis_index: int
    diagnoses: List[Dict[str, Any]]
    current_diagnosis: Optional[Dict[str, Any]]
    current_reasoning_iteration: int
    validation_queries: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    mcp_client: Optional[Any]  # Can be MCPStdioClient or MCPHTTPClient


class ProvisionalDiagnosisSystem:
    """Main system for generating provisional diagnoses."""
    
    def __init__(self):
        """Initialize the system."""
        # o1 and o3 models don't support temperature parameter - only default (1) is allowed
        # So we never set temperature for o1/o3 models
        llm_kwargs = {
            "model": MODEL_NAME,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        # Only set temperature for non-o1/o3 models (gpt-4, gpt-3.5, etc.)
        # o1-mini, o1-preview, o3 models don't accept temperature parameter
        # IMPORTANT: Do NOT include temperature in kwargs for o1/o3 models
        is_o_model = MODEL_NAME.startswith("o1") or MODEL_NAME.startswith("o3")
        if not is_o_model:
            llm_kwargs["temperature"] = 0
        
        # Create LLM - for o1/o3 models, temperature is NOT included in kwargs
        # This ensures the API doesn't receive temperature parameter for o1/o3 models
        logger.info(f"Initializing ChatOpenAI with model={MODEL_NAME}, temperature={'not set (o1/o3 model)' if is_o_model else llm_kwargs.get('temperature')}")
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Use HTTP client if server is already running, otherwise use stdio
        if USE_MCP_HTTP:
            self.mcp_client = MCPHTTPClient(MCP_HTTP_URL)
            logger.info(f"Using HTTP MCP client: {MCP_HTTP_URL}")
        else:
            self.mcp_client: Optional[MCPStdioClient] = None
            logger.info(f"Using stdio MCP client (will spawn server)")
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(DiagnosisState)
        
        # Add nodes
        workflow.add_node("query_builder", self.query_builder_node)
        workflow.add_node("initial_mcp_search", self.initial_mcp_search_node)
        workflow.add_node("diagnosis_generator", self.diagnosis_generator_node)
        workflow.add_node("validation_iterator", self.validation_iterator_node)
        workflow.add_node("diagnosis_finalizer", self.diagnosis_finalizer_node)
        workflow.add_node("output_writer", self.output_writer_node)
        
        # Set entry point
        workflow.set_entry_point("query_builder")
        
        # Add edges
        workflow.add_edge("query_builder", "initial_mcp_search")
        workflow.add_edge("initial_mcp_search", "diagnosis_generator")
        
        # Conditional edges for diagnosis loop
        workflow.add_conditional_edges(
            "diagnosis_generator",
            self.should_validate,
            {
                "validate": "validation_iterator",
                "finalize": "diagnosis_finalizer"
            }
        )
        
        # Conditional edges for validation loop
        workflow.add_conditional_edges(
            "validation_iterator",
            self.should_continue_validation,
            {
                "continue_validation": "validation_iterator",
                "finalize": "diagnosis_finalizer"
            }
        )
        
        # Conditional edges for diagnosis count
        workflow.add_conditional_edges(
            "diagnosis_finalizer",
            self.should_generate_more,
            {
                "generate_more": "diagnosis_generator",
                "output": "output_writer"
            }
        )
        
        workflow.add_edge("output_writer", END)
        
        return workflow.compile()
    
    async def query_builder_node(self, state: DiagnosisState) -> DiagnosisState:
        """Node 1: Build combined query from patient JSON."""
        logger.info("=" * 60)
        logger.info("NODE 1: Query Builder")
        logger.info("=" * 60)
        
        try:
            prompt = get_query_builder_prompt(state["patient_json"])
            response = await self.llm.ainvoke(prompt)
            
            combined_query = response.content.strip()
            logger.info(f"Generated query: {combined_query}")
            
            return {
                **state,
                "combined_query": combined_query
            }
        except Exception as e:
            logger.error(f"Error in query builder: {e}")
            return {
                **state,
                "combined_query": "patient assessment clinical details"
            }
    
    async def initial_mcp_search_node(self, state: DiagnosisState) -> DiagnosisState:
        """Node 2: Initial MCP search with combined query."""
        logger.info("=" * 60)
        logger.info("NODE 2: Initial MCP Search")
        logger.info("=" * 60)
        
        try:
            # Initialize MCP client if using stdio and not already connected
            if not USE_MCP_HTTP and (not self.mcp_client or isinstance(self.mcp_client, MCPStdioClient)):
                if not self.mcp_client:
                    self.mcp_client = MCPStdioClient(MCP_SERVER_PATH)
                    await self.mcp_client.connect()
            
            query = state["combined_query"]
            logger.info(f"Searching MCP with query: {query}")
            
            results = await self.mcp_client.search_medical_knowledge(query, top_k=10)
            # Count results if available
            result_count = 0
            if "results" in results:
                result_count = len(results.get("results", []))
            elif "text" in results or "raw_response" in results:
                # Text response, count approximate results by checking for "Result" markers
                text = results.get("text") or results.get("raw_response", "")
                result_count = text.count("Result ") or text.count("--- Result")
            logger.info(f"Received MCP search results (approx {result_count} results)")
            
            return {
                **state,
                "initial_mcp_results": results,
                "mcp_client": self.mcp_client
            }
        except Exception as e:
            logger.error(f"Error in initial MCP search: {e}")
            return {
                **state,
                "initial_mcp_results": {"error": str(e), "results": []},
                "mcp_client": self.mcp_client
            }
    
    async def diagnosis_generator_node(self, state: DiagnosisState) -> DiagnosisState:
        """Node 3: Generate one provisional diagnosis."""
        logger.info("=" * 60)
        logger.info(f"NODE 3: Diagnosis Generator (Diagnosis #{state['current_diagnosis_index'] + 1})")
        logger.info("=" * 60)
        
        try:
            previous_diagnoses = [
                diag.get("diagnosis_name", "") 
                for diag in state.get("diagnoses", [])
            ]
            
            prompt = get_initial_diagnosis_prompt(
                state["patient_json"],
                state["initial_mcp_results"],
                previous_diagnoses
            )
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                diagnosis_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON, using fallback: {e}")
                diagnosis_data = {
                    "diagnosis_name": "Unable to parse diagnosis",
                    "confidence_score": 0.5,
                    "reasoning": response_text,
                    "json_fields_used": [],
                    "validation_questions": []
                }
            
            logger.info(f"Generated diagnosis: {diagnosis_data.get('diagnosis_name', 'Unknown')}")
            
            return {
                **state,
                "current_diagnosis": diagnosis_data,
                "current_reasoning_iteration": 0,
                "validation_queries": [],
                "validation_results": []
            }
        except Exception as e:
            logger.error(f"Error in diagnosis generator: {e}")
            return {
                **state,
                "current_diagnosis": {
                    "diagnosis_name": "Error generating diagnosis",
                    "confidence_score": 0.0,
                    "reasoning": str(e),
                    "json_fields_used": [],
                    "validation_questions": []
                },
                "current_reasoning_iteration": 0,
                "validation_queries": [],
                "validation_results": []
            }
    
    async def validation_iterator_node(self, state: DiagnosisState) -> DiagnosisState:
        """Node 4: Perform validation iteration."""
        logger.info("=" * 60)
        logger.info(f"NODE 4: Validation Iterator (Iteration {state['current_reasoning_iteration'] + 1})")
        logger.info("=" * 60)
        
        try:
            current_diag = state["current_diagnosis"]
            if not current_diag:
                logger.error("No current diagnosis to validate")
                return state
            
            validation_questions = current_diag.get("validation_questions", [])
            iteration = state["current_reasoning_iteration"]
            
            if iteration >= len(validation_questions):
                logger.info("No more validation questions, proceeding to finalize")
                return {
                    **state,
                    "current_reasoning_iteration": iteration + 1
                }
            
            question = validation_questions[iteration]
            logger.info(f"Validation question: {question}")
            
            # Convert question to MCP query
            mcp_query = question  # Can be enhanced with query transformation
            
            # Search MCP
            if not USE_MCP_HTTP and (not self.mcp_client or isinstance(self.mcp_client, MCPStdioClient)):
                if not self.mcp_client:
                    self.mcp_client = MCPStdioClient(MCP_SERVER_PATH)
                    await self.mcp_client.connect()
            
            mcp_results = await self.mcp_client.search_medical_knowledge(mcp_query, top_k=5)
            
            # Get validation prompt
            previous_iterations = state.get("validation_results", [])
            prompt = get_validation_prompt(
                current_diag.get("diagnosis_name", ""),
                question,
                state["patient_json"],
                mcp_results,
                previous_iterations
            )
            
            # Get LLM validation response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                validation_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse validation JSON: {e}")
                validation_data = {
                    "validation_conclusion": response_text,
                    "confidence_adjustment": "0.0",
                    "updated_confidence": current_diag.get("confidence_score", 0.5),
                    "next_question": "CONCLUDE",
                    "evidence_summary": "Unable to parse response"
                }
            
            # Update current diagnosis confidence
            current_diag["confidence_score"] = validation_data.get("updated_confidence", current_diag.get("confidence_score", 0.5))
            
            # Store validation query and result
            validation_query = {
                "question": question,
                "mcp_query": mcp_query,
                "mcp_results": mcp_results,
                "iteration": iteration + 1,
                "validation_data": validation_data
            }
            
            validation_queries = state.get("validation_queries", [])
            validation_queries.append(validation_query)
            
            validation_results = state.get("validation_results", [])
            validation_results.append(validation_data)
            
            logger.info(f"Validation iteration {iteration + 1} complete. Confidence: {current_diag['confidence_score']}")
            
            return {
                **state,
                "current_diagnosis": current_diag,
                "current_reasoning_iteration": iteration + 1,
                "validation_queries": validation_queries,
                "validation_results": validation_results,
                "mcp_client": self.mcp_client
            }
        except Exception as e:
            logger.error(f"Error in validation iterator: {e}")
            return {
                **state,
                "current_reasoning_iteration": state["current_reasoning_iteration"] + 1
            }
    
    async def diagnosis_finalizer_node(self, state: DiagnosisState) -> DiagnosisState:
        """Node 5: Finalize diagnosis with complete reasoning."""
        logger.info("=" * 60)
        logger.info("NODE 5: Diagnosis Finalizer")
        logger.info("=" * 60)
        
        try:
            current_diag = state["current_diagnosis"]
            if not current_diag:
                logger.error("No current diagnosis to finalize")
                return state
            
            all_validation_iterations = state.get("validation_results", [])
            
            prompt = get_final_diagnosis_prompt(
                current_diag.get("diagnosis_name", ""),
                state["patient_json"],
                all_validation_iterations,
                state["initial_mcp_results"]
            )
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                final_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse final diagnosis JSON: {e}")
                final_data = {
                    "diagnosis_name": current_diag.get("diagnosis_name", "Unknown"),
                    "confidence_score": current_diag.get("confidence_score", 0.5),
                    "reasoning": response_text,
                    "json_fields_used": current_diag.get("json_fields_used", []),
                    "supporting_evidence": []
                }
            
            # Build ValidationQuery objects
            validation_queries = []
            for vq in state.get("validation_queries", []):
                validation_queries.append(ValidationQuery(
                    question=vq["question"],
                    mcp_query=vq["mcp_query"],
                    mcp_results=vq["mcp_results"],
                    iteration=vq["iteration"]
                ).dict())
            
            # Build final DiagnosisResult
            diagnosis_result = DiagnosisResult(
                diagnosis_name=final_data.get("diagnosis_name", "Unknown"),
                confidence_score=float(final_data.get("confidence_score", 0.5)),
                reasoning=final_data.get("reasoning", ""),
                json_fields_used=final_data.get("json_fields_used", []),
                supporting_evidence=final_data.get("supporting_evidence", []),
                validation_queries=validation_queries,
                iteration_count=len(validation_queries)
            )
            
            # Add to diagnoses list
            diagnoses = state.get("diagnoses", [])
            diagnoses.append(diagnosis_result.dict())
            
            logger.info(f"Finalized diagnosis: {diagnosis_result.diagnosis_name}")
            
            return {
                **state,
                "diagnoses": diagnoses,
                "current_diagnosis_index": state["current_diagnosis_index"] + 1,
                "current_diagnosis": None,
                "current_reasoning_iteration": 0,
                "validation_queries": [],
                "validation_results": []
            }
        except Exception as e:
            logger.error(f"Error in diagnosis finalizer: {e}")
            return state
    
    async def output_writer_node(self, state: DiagnosisState) -> DiagnosisState:
        """Node 6: Write output to file."""
        logger.info("=" * 60)
        logger.info("NODE 6: Output Writer")
        logger.info("=" * 60)
        
        try:
            patient_id = state["patient_id"]
            output_dir = Path(OUTPUT_DIR) / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build final report
            final_report = FinalReport(
                patient_id=patient_id,
                appointment_id=state["patient_json"].get("appointment"),
                diagnoses=[DiagnosisResult(**diag) for diag in state["diagnoses"]],
                combined_query=state["combined_query"],
                initial_mcp_results=state["initial_mcp_results"]
            )
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"diagnosis_report_{timestamp}.json"
            
            with open(output_file, "w") as f:
                json.dump(final_report.dict(), f, indent=2)
            
            logger.info(f"✓ Report saved to: {output_file}")
            logger.info(f"✓ Generated {len(state['diagnoses'])} diagnoses")
            
            return state
        except Exception as e:
            logger.error(f"Error in output writer: {e}")
            return state
    
    def should_validate(self, state: DiagnosisState) -> str:
        """Determine if validation is needed."""
        current_diag = state.get("current_diagnosis")
        if not current_diag:
            return "finalize"
        
        validation_questions = current_diag.get("validation_questions", [])
        if len(validation_questions) > 0:
            return "validate"
        else:
            return "finalize"
    
    def should_continue_validation(self, state: DiagnosisState) -> str:
        """Determine if validation should continue."""
        iteration = state.get("current_reasoning_iteration", 0)
        current_diag = state.get("current_diagnosis", {})
        validation_questions = current_diag.get("validation_questions", [])
        
        # Check if we've reached max iterations
        if iteration >= MAX_ITERATIONS_PER_DIAGNOSIS:
            logger.info(f"Reached max iterations ({MAX_ITERATIONS_PER_DIAGNOSIS}), finalizing")
            return "finalize"
        
        # Check if we've processed all questions
        if iteration >= len(validation_questions):
            return "finalize"
        
        # Check if last validation said to conclude
        validation_results = state.get("validation_results", [])
        if validation_results:
            last_result = validation_results[-1]
            next_question = last_result.get("next_question", "")
            if next_question.upper() == "CONCLUDE":
                logger.info("Validation concluded by LLM")
                return "finalize"
        
        return "continue_validation"
    
    def should_generate_more(self, state: DiagnosisState) -> str:
        """Determine if more diagnoses should be generated."""
        current_index = state.get("current_diagnosis_index", 0)
        if current_index < NUM_DIAGNOSES:
            return "generate_more"
        else:
            return "output"
    
    async def process_patient_json(self, json_path: str) -> FinalReport:
        """Process a patient JSON file and generate diagnoses."""
        logger.info("=" * 60)
        logger.info("PROVISIONAL DIAGNOSIS SYSTEM")
        logger.info("=" * 60)
        
        # Load JSON
        with open(json_path, "r") as f:
            patient_json = json.load(f)
        
        # Extract patient ID
        patient_id = Path(json_path).stem
        
        # Initialize state
        initial_state: DiagnosisState = {
            "patient_json": patient_json,
            "patient_id": patient_id,
            "combined_query": "",
            "initial_mcp_results": {},
            "current_diagnosis_index": 0,
            "diagnoses": [],
            "current_diagnosis": None,
            "current_reasoning_iteration": 0,
            "validation_queries": [],
            "validation_results": [],
            "mcp_client": None
        }
        
        # Run graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            # Build final report
            final_report = FinalReport(
                patient_id=patient_id,
                appointment_id=patient_json.get("appointment"),
                diagnoses=[DiagnosisResult(**diag) for diag in final_state["diagnoses"]],
                combined_query=final_state["combined_query"],
                initial_mcp_results=final_state["initial_mcp_results"]
            )
            
            return final_report
        finally:
            # Cleanup MCP client (only for stdio, HTTP doesn't need cleanup)
            if self.mcp_client and isinstance(self.mcp_client, MCPStdioClient):
                await self.mcp_client.disconnect()


# ---- Flask Server Implementation ----
import threading

# Create Flask app (only if Flask is available)
if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)  # Enable CORS for cross-origin requests
else:
    app = None

# Global system instance
diagnosis_system: Optional[ProvisionalDiagnosisSystem] = None
system_lock = threading.Lock()


def get_system() -> ProvisionalDiagnosisSystem:
    """Get or create the diagnosis system instance."""
    global diagnosis_system
    with system_lock:
        if diagnosis_system is None:
            logger.info("Initializing Provisional Diagnosis System...")
            diagnosis_system = ProvisionalDiagnosisSystem()
            logger.info("✓ System initialized")
        return diagnosis_system


if FLASK_AVAILABLE:
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "running",
            "service": "provisional_diagnosis_system",
            "model": MODEL_NAME,
            "max_iterations": MAX_ITERATIONS_PER_DIAGNOSIS,
            "num_diagnoses": NUM_DIAGNOSES
        })

    @app.route('/diagnose', methods=['POST'])
    def diagnose_endpoint():
        """Main endpoint for generating provisional diagnoses."""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # Accept either file path or JSON data directly
            json_path = data.get("json_path")
            assessment_json = data.get("assessment_json")
            
            if not json_path and not assessment_json:
                return jsonify({"error": "Either 'json_path' or 'assessment_json' must be provided"}), 400
            
            logger.info("=" * 60)
            logger.info("DIAGNOSIS REQUEST RECEIVED")
            logger.info("=" * 60)
            
            # Get system instance
            system = get_system()
            
            # Process in async context
            if json_path:
                # Process from file path
                logger.info(f"Processing file: {json_path}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    report = loop.run_until_complete(
                        system.process_patient_json(json_path)
                    )
                finally:
                    loop.close()
            else:
                # Process from JSON data directly
                logger.info("Processing JSON data directly")
                import tempfile
                import uuid
                
                # Create temporary file
                temp_id = str(uuid.uuid4())
                temp_file = Path(tempfile.gettempdir()) / f"assessment_{temp_id}.json"
                
                try:
                    with open(temp_file, "w") as f:
                        json.dump(assessment_json, f)
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        report = loop.run_until_complete(
                            system.process_patient_json(str(temp_file))
                        )
                    finally:
                        loop.close()
                finally:
                    # Clean up temp file
                    if temp_file.exists():
                        temp_file.unlink()
            
            # Convert report to dict for JSON response
            report_dict = report.dict()
            
            logger.info(f"✓ Diagnosis complete: {len(report.diagnoses)} diagnoses generated")
            
            return jsonify({
                "status": "success",
                "report": report_dict
            })
            
        except Exception as e:
            logger.error(f"Error in diagnose endpoint: {e}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            return jsonify({
                "status": "error",
                "error": str(e),
                "traceback": error_trace
            }), 500


    def run_server(host='0.0.0.0', port=5051, debug=False):
        """Run the Flask server."""
        if not FLASK_AVAILABLE:
            logger.error("Flask is not available. Cannot run server.")
            logger.error("Please install: pip install flask flask-cors")
            sys.exit(1)
        
        logger.info("=" * 60)
        logger.info("PROVISIONAL DIAGNOSIS SERVER")
        logger.info("=" * 60)
        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Max iterations per diagnosis: {MAX_ITERATIONS_PER_DIAGNOSIS}")
        logger.info(f"Number of diagnoses: {NUM_DIAGNOSES}")
        logger.info("=" * 60)
        logger.info("Endpoints:")
        logger.info("  GET  /health - Health check")
        logger.info("  POST /diagnose - Generate provisional diagnoses")
        logger.info("=" * 60)
        
        app.run(host=host, port=port, debug=debug, threaded=True)
else:
    def run_server(host='0.0.0.0', port=5051, debug=False):
        """Placeholder when Flask is not available."""
        logger.error("Flask is not available. Cannot run server.")
        logger.error("Please install: pip install flask flask-cors")
        sys.exit(1)


async def main_cli():
    """CLI entry point (for direct execution)."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate provisional diagnoses from patient JSON")
    parser.add_argument("--json", required=True, help="Path to patient assessment JSON file")
    parser.add_argument("--server", action="store_true", help="Run as server instead of CLI")
    parser.add_argument("--port", type=int, default=5051, help="Server port (default: 5051)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()
    
    if args.server:
        # Run as server
        run_server(host=args.host, port=args.port)
    else:
        # Run as CLI
        system = ProvisionalDiagnosisSystem()
        
        try:
            report = await system.process_patient_json(args.json)
            print("\n" + "=" * 60)
            print("DIAGNOSIS COMPLETE")
            print("=" * 60)
            print(f"Patient ID: {report.patient_id}")
            print(f"Generated {len(report.diagnoses)} diagnoses:")
            for i, diag in enumerate(report.diagnoses, 1):
                print(f"\n{i}. {diag.diagnosis_name} (Confidence: {diag.confidence_score:.2f})")
            print("\n" + "=" * 60)
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    # Check if running as server or CLI
    if "--server" in sys.argv or len(sys.argv) == 1:
        # Run as server
        port = 5051
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        
        host = "0.0.0.0"
        if "--host" in sys.argv:
            idx = sys.argv.index("--host")
            if idx + 1 < len(sys.argv):
                host = sys.argv[idx + 1]
        
        run_server(host=host, port=port)
    else:
        # Run as CLI
        asyncio.run(main_cli())

