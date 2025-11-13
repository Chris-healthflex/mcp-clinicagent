"""
Prompt templates for provisional diagnosis system
"""

from typing import List, Dict, Any


def get_query_builder_prompt(patient_json: Dict[str, Any]) -> str:
    """Generate prompt for building combined query from patient JSON."""
    return f"""You are a medical query builder. Analyze the following patient assessment JSON and create a comprehensive search query that captures all relevant clinical information.

Patient Assessment Data:
{format_patient_data(patient_json)}

Your task:
1. Extract key clinical information from:
   - Chief complaints
   - Client history
   - Objective assessment tests and values
   - Subjective assessment notes
   - Any relevant clinical details

2. Create a single, comprehensive search query (2-3 sentences) that combines:
   - Primary symptoms/complaints
   - Relevant test findings
   - Clinical context (e.g., post-operative status, duration)
   - Key objective measurements

3. The query should be optimized for medical knowledge base search.

Output ONLY the search query text, nothing else."""


def get_initial_diagnosis_prompt(
    patient_json: Dict[str, Any],
    mcp_results: Dict[str, Any],
    previous_diagnoses: List[str]
) -> str:
    """Generate prompt for initial diagnosis generation."""
    previous_diag_text = ""
    if previous_diagnoses:
        previous_diag_text = "\n\nPrevious diagnoses already generated (DO NOT repeat these):\n"
        for i, diag in enumerate(previous_diagnoses, 1):
            previous_diag_text += f"{i}. {diag}\n"
    
    return f"""You are an expert medical diagnostician. Analyze the patient assessment and medical knowledge base results to generate ONE provisional diagnosis.

Patient Assessment:
{format_patient_data(patient_json)}

Medical Knowledge Base Results:
{format_mcp_results(mcp_results)}
{previous_diag_text}

Your task:
1. Generate ONE distinct provisional diagnosis that is different from previous diagnoses
2. Consider the clinical presentation, test results, and medical knowledge
3. Generate 3-5 internal reasoning questions you need to validate (e.g., "Is this consistent with the test findings?", "Are there other conditions with similar presentation?")
4. Identify which specific JSON fields you are using for this diagnosis

Output your response in this EXACT JSON format:
{{
    "diagnosis_name": "Provisional diagnosis name",
    "confidence_score": 0.0-1.0,
    "reasoning": "Brief initial reasoning",
    "json_fields_used": ["field.path.1", "field.path.2"],
    "validation_questions": [
        "Question 1 to validate",
        "Question 2 to validate",
        "Question 3 to validate"
    ]
}}

Be specific and concise. Use medical terminology appropriate for the case."""


def get_validation_prompt(
    diagnosis_name: str,
    validation_question: str,
    patient_json: Dict[str, Any],
    mcp_validation_results: Dict[str, Any],
    previous_iterations: List[Dict[str, Any]]
) -> str:
    """Generate prompt for validation iteration."""
    prev_iter_text = ""
    if previous_iterations:
        prev_iter_text = "\n\nPrevious validation iterations:\n"
        for i, iter_data in enumerate(previous_iterations, 1):
            prev_iter_text += f"Iteration {i}: {iter_data.get('question', 'N/A')}\n"
            prev_iter_text += f"  Result: {iter_data.get('conclusion', 'N/A')}\n"
    
    return f"""You are validating a provisional diagnosis through iterative reasoning.

Provisional Diagnosis: {diagnosis_name}

Current Validation Question: {validation_question}

Patient Assessment:
{format_patient_data(patient_json)}

Medical Knowledge Base Results for this question:
{format_mcp_results(mcp_validation_results)}
{prev_iter_text}

Your task:
1. Analyze the validation question in context of the diagnosis
2. Review the medical knowledge base results
3. Determine if the evidence supports or refutes the diagnosis
4. Refine your confidence in the diagnosis
5. Generate the next validation question if needed (or conclude if confident)

Output your response in this EXACT JSON format:
{{
    "validation_conclusion": "What you learned from this validation",
    "confidence_adjustment": "+0.1 or -0.1 or 0.0",
    "updated_confidence": 0.0-1.0,
    "next_question": "Next validation question OR 'CONCLUDE' if confident enough",
    "evidence_summary": "Key evidence points from this iteration"
}}

Be analytical and evidence-based."""


def get_final_diagnosis_prompt(
    diagnosis_name: str,
    patient_json: Dict[str, Any],
    all_validation_iterations: List[Dict[str, Any]],
    initial_mcp_results: Dict[str, Any]
) -> str:
    """Generate prompt for finalizing diagnosis with complete reasoning."""
    validation_summary = "\n".join([
        f"Iteration {i+1}: {iter_data.get('question', 'N/A')}\n"
        f"  Conclusion: {iter_data.get('validation_conclusion', 'N/A')}\n"
        f"  Evidence: {iter_data.get('evidence_summary', 'N/A')}"
        for i, iter_data in enumerate(all_validation_iterations)
    ])
    
    return f"""You are finalizing a provisional diagnosis with complete reasoning.

Provisional Diagnosis: {diagnosis_name}

Patient Assessment:
{format_patient_data(patient_json)}

Initial Medical Knowledge Base Results:
{format_mcp_results(initial_mcp_results)}

Validation Iterations:
{validation_summary}

Your task:
1. Synthesize all information into a final, well-reasoned diagnosis
2. Provide a comprehensive explanation of:
   - Why this diagnosis fits the clinical presentation
   - How you reached this conclusion
   - Which specific JSON fields were most important
   - Supporting evidence from the medical knowledge base
3. Provide a final confidence score (0.0-1.0)
4. List all JSON field paths used

Output your response in this EXACT JSON format:
{{
    "diagnosis_name": "Final provisional diagnosis name",
    "confidence_score": 0.0-1.0,
    "reasoning": "Comprehensive explanation of why and how you reached this diagnosis, including key clinical findings, test results, and medical knowledge that supports it",
    "json_fields_used": ["complete.list.of.field.paths"],
    "supporting_evidence": [
        {{"source": "source_name", "relevance": "how it supports diagnosis", "content": "key excerpt"}}
    ]
}}

Be thorough, evidence-based, and clear."""


def format_patient_data(patient_json: Dict[str, Any]) -> str:
    """Format patient JSON data for prompts."""
    records = patient_json.get("records", {})
    clinical = records.get("clinicalDetails", {})
    objective = records.get("objectiveAssessment", {})
    subjective = records.get("subjectiveAssessment", {})
    
    formatted = "=== CLINICAL DETAILS ===\n"
    formatted += f"Chief Complaints: {clinical.get('chiefComplaints', 'N/A')}\n"
    formatted += f"Client History: {clinical.get('clientHistory', 'N/A')}\n"
    formatted += f"Duration: {clinical.get('duration', 'N/A')}\n"
    
    formatted += "\n=== OBJECTIVE ASSESSMENT ===\n"
    tests = objective.get("tests", [])
    for test in tests:
        test_name = test.get("testName", "Unknown")
        value = test.get("value", "N/A")
        unit = test.get("unitName", "")
        left = test.get("left", None)
        right = test.get("right", None)
        
        if left is not None and right is not None:
            formatted += f"{test_name}: L={left}{unit}, R={right}{unit}\n"
        else:
            formatted += f"{test_name}: {value}{unit}\n"
    
    formatted += "\n=== SUBJECTIVE ASSESSMENT ===\n"
    formatted += f"{subjective.get('assessment', 'N/A')}\n"
    
    return formatted


def format_mcp_results(mcp_results: Dict[str, Any]) -> str:
    """Format MCP search results for prompts."""
    if not mcp_results:
        return "No results found."
    
    if "error" in mcp_results:
        return f"Error occurred: {mcp_results.get('error', 'Unknown error')}"
    
    # Handle different MCP result formats
    if "text" in mcp_results or "raw_response" in mcp_results:
        # Direct text response from MCP server
        text = mcp_results.get("text") or mcp_results.get("raw_response", "")
        # The MCP server returns formatted text, use it directly
        return text
    elif "results" in mcp_results:
        # Structured results (if server returns JSON)
        results = mcp_results.get("results", [])
        if not results:
            return "No results found."
        
        formatted = f"Found {len(results)} results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"Result {i} (Relevance: {result.get('relevance_score', 'N/A')}):\n"
            formatted += f"Source: {result.get('source', 'N/A')}\n"
            content = result.get('content', '')
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            formatted += f"{content}\n\n"
        return formatted
    else:
        # Fallback: return string representation
        return str(mcp_results)

