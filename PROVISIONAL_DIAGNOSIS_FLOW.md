# Provisional Diagnosis System - Flow Diagram

```mermaid
flowchart TD
    Start([Start: Patient JSON Input]) --> QueryBuilder[Node 1: Query Builder<br/>LLM analyzes JSON<br/>Generates combined query]
    
    QueryBuilder --> MCPSearch[Node 2: Initial MCP Search<br/>Search medical knowledge base<br/>with combined query]
    
    MCPSearch --> DiagGen[Node 3: Diagnosis Generator<br/>LLM generates ONE diagnosis<br/>+ 3-5 validation questions]
    
    DiagGen --> CheckValidate{Has validation<br/>questions?}
    
    CheckValidate -->|Yes| ValidationIter[Node 4: Validation Iterator<br/>Loop: For each question<br/>1. Convert to MCP query<br/>2. Search knowledge base<br/>3. LLM validates evidence<br/>4. Update confidence<br/>5. Generate next question]
    
    ValidationIter --> CheckContinue{Continue<br/>validation?}
    
    CheckContinue -->|Yes<br/>More questions| ValidationIter
    CheckContinue -->|No<br/>CONCLUDE or max iterations| DiagFinalizer
    
    CheckValidate -->|No| DiagFinalizer[Node 5: Diagnosis Finalizer<br/>LLM synthesizes all evidence<br/>Creates final diagnosis with<br/>complete reasoning]
    
    DiagFinalizer --> CheckMore{More diagnoses<br/>needed?<br/>Current < 5}
    
    CheckMore -->|Yes<br/>Generate more| DiagGen
    CheckMore -->|No<br/>5 diagnoses done| OutputWriter[Node 6: Output Writer<br/>Save final report to JSON<br/>provisional_users/patient_id/]
    
    OutputWriter --> End([End: Report Saved])
    
    %% External components
    MCPSearch -.->|HTTP/STDIO| MCPServer[(MCP Medical Server<br/>ChromaDB Knowledge Base)]
    ValidationIter -.->|HTTP/STDIO| MCPServer
    QueryBuilder -.->|API Call| LLM[(OpenAI o3/o1-mini<br/>Reasoning Model)]
    DiagGen -.->|API Call| LLM
    ValidationIter -.->|API Call| LLM
    DiagFinalizer -.->|API Call| LLM
    
    %% Styling
    classDef nodeStyle fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef decisionStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef externalStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef startEndStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px
    
    class QueryBuilder,MCPSearch,DiagGen,ValidationIter,DiagFinalizer,OutputWriter nodeStyle
    class CheckValidate,CheckContinue,CheckMore decisionStyle
    class MCPServer,LLM externalStyle
    class Start,End startEndStyle
```

## System Overview

### Main Flow
1. **Query Builder** - Analyzes patient JSON and creates a comprehensive search query
2. **Initial MCP Search** - Searches medical knowledge base with the combined query
3. **Diagnosis Generator** - Generates ONE provisional diagnosis with 3-5 validation questions
4. **Validation Iterator** - Loops through validation questions (3-5 iterations per diagnosis):
   - Converts question to MCP query
   - Searches medical knowledge base
   - LLM validates evidence and updates confidence
   - Generates next question or concludes
5. **Diagnosis Finalizer** - Synthesizes all evidence into final diagnosis with complete reasoning
6. **Output Writer** - Saves final report to JSON file

### Loops
- **Validation Loop**: Iterates 3-5 times per diagnosis (or until LLM says "CONCLUDE")
- **Diagnosis Loop**: Generates 5 different diagnoses total

### External Components
- **MCP Medical Server**: ChromaDB-based medical knowledge base (accessed via HTTP or STDIO)
- **OpenAI LLM**: o3 or o1-mini reasoning model for all LLM operations

### Output
Final report saved to: `provisional_users/{patient_id}/diagnosis_report_{timestamp}.json`

