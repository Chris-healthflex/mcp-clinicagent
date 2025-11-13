"""
Pydantic models for provisional diagnosis system
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ValidationQuery(BaseModel):
    """Represents a validation query and its results."""
    question: str = Field(description="The reasoning question being asked")
    mcp_query: str = Field(description="The query sent to MCP")
    mcp_results: Dict[str, Any] = Field(description="Results from MCP search")
    iteration: int = Field(description="Iteration number (1-5)")


class DiagnosisResult(BaseModel):
    """Represents a single provisional diagnosis with full reasoning."""
    diagnosis_name: str = Field(description="Name of the provisional diagnosis")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(description="Detailed reasoning explanation")
    json_fields_used: List[str] = Field(description="List of JSON field paths used")
    supporting_evidence: List[Dict[str, Any]] = Field(
        description="Supporting evidence from MCP searches"
    )
    validation_queries: List[ValidationQuery] = Field(
        description="Validation queries and results used"
    )
    iteration_count: int = Field(description="Number of validation iterations performed")


class FinalReport(BaseModel):
    """Final report containing all diagnoses."""
    patient_id: str = Field(description="Patient ID from JSON")
    appointment_id: Optional[str] = Field(None, description="Appointment ID if available")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    diagnoses: List[DiagnosisResult] = Field(description="List of 5 provisional diagnoses")
    combined_query: str = Field(description="Initial combined query used")
    initial_mcp_results: Dict[str, Any] = Field(description="Initial MCP search results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "R-DEF-9Q31229",
                "diagnoses": [],
                "combined_query": "knee pain post ACL reconstruction",
                "initial_mcp_results": {}
            }
        }

