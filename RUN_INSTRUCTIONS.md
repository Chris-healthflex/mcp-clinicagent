# How to Run the Provisional Diagnosis System

## Architecture

The system now runs as a **server-client architecture**:
- **Server**: `provisional_diagnosis_system.py` runs as a Flask server
- **Client**: `medical_chat.py` connects to the server via HTTP

## Prerequisites

1. **Install Dependencies**
   ```bash
   cd /Users/chrisdev/Healthflex/mcp-clinicagent
   pip install -r requirements.txt
   pip install mcp chromadb sentence-transformers simple-term-menu flask-cors
   ```

2. **Set Up Environment Variables**
   Create or update `.env` file with:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL_NAME=o1-mini
   MCP_SERVER_PATH=./mcp_medical_server.py
   DIAGNOSIS_OUTPUT_DIR=./provisional_users
   MAX_ITERATIONS_PER_DIAGNOSIS=5
   NUM_DIAGNOSES=5
   PROVISIONAL_SERVER_URL=http://localhost:5051  # Optional, defaults to this
   ```

## Running the System

### Step 1: Start the Server

In **Terminal 1**, start the provisional diagnosis server:

```bash
cd /Users/chrisdev/Healthflex/mcp-clinicagent
python provisional_diagnosis_system.py --server
```

Or with custom port:
```bash
python provisional_diagnosis_system.py --server --port 5051
```

You should see:
```
============================================================
PROVISIONAL DIAGNOSIS SERVER
============================================================
Starting server on 0.0.0.0:5051
Model: o1-mini
Max iterations per diagnosis: 5
Number of diagnoses: 5
============================================================
Endpoints:
  GET  /health - Health check
  POST /diagnose - Generate provisional diagnoses
============================================================
```

**Keep this terminal running!**

### Step 2: Start the Chat Client

In **Terminal 2**, start the chat interface:

```bash
cd /Users/chrisdev/Healthflex/mcp-clinicagent
python medical_chat.py
```

### Step 3: Use the Provisional Diagnosis System

Once the chat interface starts:

1. **Run the command:**
   ```
   /provisional
   ```

2. **Select a patient file** from the menu (use arrow keys, press Enter)

3. **Wait for processing** (5-15 minutes depending on complexity)

4. **View results** in the terminal and saved JSON file

## Alternative: Direct CLI Usage

You can also run the system directly without the server:

```bash
python provisional_diagnosis_system.py --json dec-tree/first_assessments/R-DEF-9Q31229.json
```

## Server Endpoints

### Health Check
```bash
curl http://localhost:5051/health
```

### Diagnose (via file path)
```bash
curl -X POST http://localhost:5051/diagnose \
  -H "Content-Type: application/json" \
  -d '{"json_path": "/path/to/assessment.json"}'
```

### Diagnose (via JSON data)
```bash
curl -X POST http://localhost:5051/diagnose \
  -H "Content-Type: application/json" \
  -d '{"assessment_json": {...}}'
```

## Output

Results are saved to:
```
provisional_users/{patient_id}/diagnosis_report_{timestamp}.json
```

For example:
```
provisional_users/R-DEF-9Q31229/diagnosis_report_20241220_143022.json
```

## Troubleshooting

### "Provisional diagnosis server is not running"
- Make sure you started the server in Terminal 1
- Check the server is running: `curl http://localhost:5051/health`
- Verify the port matches: default is 5051

### "Connection refused"
- Check if the server is actually running
- Verify the port number matches
- Check firewall settings

### "MCP server connection failed"
- Ensure `mcp_medical_server.py` exists and is executable
- Check that ChromaDB database exists at `./chroma_db_collective`
- Verify Python path: `which python3`

### "OpenAI API key not found"
- Check `.env` file has `OPENAI_API_KEY=your_key`
- Or set environment variable: `export OPENAI_API_KEY=your_key`

## Notes

- **Server must be running** before using the chat interface
- Processing time: 5-15 minutes per patient
- Each diagnosis goes through 3-5 validation iterations
- The system uses o1-mini for reasoning
- MCP server runs in the background via stdio protocol
- Server runs on port 5051 by default (configurable)
