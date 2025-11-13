#!/usr/bin/env python3
"""
Interactive Medical Chat Terminal
Chat interface to interact with the LangChain Medical API Server with MCP Tools
"""

import requests
import json
import sys
import os
import asyncio
from datetime import datetime
import time
from pathlib import Path
try:
    from simple_term_menu import TerminalMenu
except ImportError:
    print("Installing required package: simple-term-menu")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "simple-term-menu"])
    from simple_term_menu import TerminalMenu

# Provisional diagnosis server configuration
PROVISIONAL_SERVER_URL = os.getenv("PROVISIONAL_SERVER_URL", "http://localhost:5051")
PROVISIONAL_DIAGNOSIS_AVAILABLE = True  # Always available via HTTP

class MedicalChatInterface:
    """Interactive chat interface for the medical API."""
    
    def __init__(self, base_url="http://localhost:5050"):
        self.base_url = base_url
        self.session_history = []
        self.current_mode = "intelligent"  # intelligent, retrieval, direct
        self.assessments_dir = Path("/Users/chrisdev/Healthflex/mcp-clinicagent/dec-tree/first_assessments")
        self.outputs_dir = Path("/Users/chrisdev/Healthflex/mcp-clinicagent/provisional_users")
        self.selected_file = None
        
        # Available modes
        self.modes = {
            "intelligent": {
                "endpoint": "/agent_ask",
                "description": " Optimized LangChain Agent (Recommended)",
                "color": "\033[94m"  # Blue
            },
            "retrieval": {
                "endpoint": "/retrieval_ask", 
                "description": " Fast RetrievalQA",
                "color": "\033[92m"  # Green
            },
            "direct": {
                "endpoint": "/fast_ask",
                "description": " Ultra-Fast Direct Search",
                "color": "\033[93m"  # Yellow
            },
            "langgraph": {
                "endpoint": "/langgraph_diagnose",
                "description": "🧠 LangGraph + LangChain Reasoning Agent (Multi-Query, 5 Diagnoses)",
                "color": "\033[95m"  # Purple
            },
            "provisional": {
                "endpoint": None,  # Direct execution, no API endpoint
                "description": "🔬 Provisional Diagnosis System (o1-mini, 5 Diagnoses, MCP Validation)",
                "color": "\033[96m"  # Cyan
            }
        }
        
        # Colors
        self.colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "purple": "\033[95m",
            "cyan": "\033[96m"
        }
    
    def print_header(self):
        """Print the chat interface header."""
        # ASCII Art Banner
        print(f"{self.colors['bold']}{self.colors['blue']}")
        print("=" * 80)
        
        # Big ASCII text using dashes and characters
        banner_text = [
            "███╗   ███╗███████╗██████╗ ██╗ ██████╗ █████╗ ██╗          █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
            "████╗ ████║██╔════╝██╔══██╗██║██╔════╝██╔══██╗██║         ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
            "██╔████╔██║█████╗  ██║  ██║██║██║     ███████║██║         ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ",
            "██║╚██╔╝██║██╔══╝  ██║  ██║██║██║     ██╔══██║██║         ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ",
            "██║ ╚═╝ ██║███████╗██████╔╝██║╚██████╗██║  ██║███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ",
            "╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   "
        ]
        
        for line in banner_text:
            print(line)
        
        print("=" * 80)
        print(" Connected to LangChain Medical API with MCP Tools")
        print("=" * 80)
        print(f"{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}Available Modes:{self.colors['reset']}")
        for mode, info in self.modes.items():
            marker = "." if mode == self.current_mode else "  "
            print(f"{marker} {mode}: {info['description']}")
        
        print(f"\n{self.colors['yellow']}Commands:{self.colors['reset']}")
        print("  /mode <mode_name>  - Switch between modes")
        print("  /file             - Analyze a patient assessment file")
        print("  /provisional      - Run provisional diagnosis system (o1-mini + MCP)")
        print("  /batch            - Batch process all assessment files")
        print("  /history          - View conversation history") 
        print("  /clear            - Clear conversation history")
        print("  /health           - Check API server status")
        print("  /help             - Show this help")
        print("  /quit or /exit    - Exit the chat")
        print("=" * 80)
    
    def check_server_health(self, silent=False):
        """Check if the API server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                if not silent:
                    health_data = response.json()
                    print(f"{self.colors['green']}✓ Medical API Server (port 5050): {health_data.get('status', 'unknown')}{self.colors['reset']}")
                    print(f"   MCP Status: {health_data.get('mcp_status', 'unknown')}")
                    print(f"   Database: {health_data.get('database_path', 'unknown')}")
                return True
            else:
                return False
        except requests.exceptions.RequestException:
            return False
    
    def send_question(self, question):
        """Send question to the appropriate API endpoint."""
        mode_info = self.modes[self.current_mode]
        
        # Handle provisional diagnosis mode (direct execution)
        if self.current_mode == "provisional" or mode_info.get("endpoint") is None:
            print(f"{self.colors['red']} Provisional diagnosis mode requires a JSON file.{self.colors['reset']}")
            print(f"{self.colors['yellow']} Use /provisional or /file to select a patient assessment file.{self.colors['reset']}")
            return
        
        # Check if server is available for non-provisional modes
        if not self.check_server_health():
            print(f"{self.colors['red']} Medical API server (port 5050) is not running.{self.colors['reset']}")
            print(f"{self.colors['yellow']} Please start it with: python Langapproach.py{self.colors['reset']}")
            print(f"{self.colors['yellow']} Or use /provisional mode which uses a different server.{self.colors['reset']}")
            return
        
        endpoint = self.base_url + mode_info["endpoint"]
        
        try:
            print(f"{self.colors['yellow']} Processing with {mode_info['description']}...{self.colors['reset']}")
            
            # Prepare request based on mode
            if self.current_mode == "direct":
                # For direct mode, use the same format as other modes
                payload = {"question": question}
            else:
                payload = {"question": question}
            
            # Send request (longer timeout for langgraph mode)
            start_time = time.time()
            timeout = 600 if self.current_mode == "langgraph" else 120
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.display_response(result, response_time)
                
                # Save to history
                self.session_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "question": question,
                    "mode": self.current_mode,
                    "response": result.get("answer", result.get("mcp_result", "No answer")),
                    "response_time": response_time
                })
                
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                print(f"{self.colors['red']} Error {response.status_code}: {error_data.get('error', 'Unknown error')}{self.colors['reset']}")
                
        except requests.exceptions.Timeout:
            print(f"{self.colors['red']} Request timed out. The query might be complex and taking longer to process.{self.colors['reset']}")
        except requests.exceptions.RequestException as e:
            print(f"{self.colors['red']} Request failed: {e}{self.colors['reset']}")
        except json.JSONDecodeError:
            print(f"{self.colors['red']} Invalid JSON response from server{self.colors['reset']}")
    
    def display_response(self, result, response_time):
        """Display the API response in a formatted way."""
        print(f"\n{self.colors['green']}{self.colors['bold']}🤖 MEDICAL ASSISTANT RESPONSE:{self.colors['reset']}")
        print("=" * 60)
        
        # Handle langgraph format (diagnoses list)
        if "diagnoses" in result:
            print(f"{self.colors['cyan']}Generated {len(result['diagnoses'])} diagnoses:{self.colors['reset']}\n")
            for i, diag in enumerate(result["diagnoses"], 1):
                print(f"{self.colors['bold']}{i}. {diag.get('diagnosis', 'N/A')}{self.colors['reset']}")
                print(f"   Probability: {diag.get('probability', 0)}%")
                if diag.get('grounding'):
                    print(f"   Grounding:")
                    for j, evidence in enumerate(diag['grounding'][:2], 1):  # Show first 2
                        print(f"     - {evidence[:150]}...")
                if diag.get('required_fields'):
                    print(f"   Fields used: {', '.join(diag['required_fields'][:3])}")
                print()
        else:
            # Main answer (for other modes)
            answer = result.get("answer", result.get("mcp_result", "No answer provided"))
            print(f"{answer}\n")
        
        # Additional information
        if "sources" in result and result["sources"]:
            print(f"{self.colors['cyan']} Sources:{self.colors['reset']}")
            for i, source in enumerate(result["sources"][:3], 1):  # Show top 3 sources
                if isinstance(source, dict):
                    source_info = source.get("source", "Unknown source")
                    similarity = source.get("similarity", 0)
                    print(f"  {i}. {source_info} (Relevance: {similarity:.3f})")
                else:
                    print(f"  {i}. {source}")
            print()
        
        # Workflow information
        if "workflow" in result:
            print(f"{self.colors['purple']} Workflow: {result['workflow']}{self.colors['reset']}")
        
        if "approach" in result:
            print(f"{self.colors['purple']} Approach: {result['approach']}{self.colors['reset']}")
        
        # Response time
        print(f"{self.colors['yellow']} Response time: {response_time:.2f}s{self.colors['reset']}")
        print("=" * 60)
    
    def handle_command(self, command):
        """Handle special commands."""
        parts = command.strip().lower().split()
        cmd = parts[0]
        
        if cmd in ["/quit", "/exit"]:
            return False
        
        elif cmd == "/mode":
            if len(parts) > 1 and parts[1] in self.modes:
                self.current_mode = parts[1]
                mode_info = self.modes[self.current_mode]
                print(f"{self.colors['green']} Switched to: {mode_info['description']}{self.colors['reset']}")
            else:
                print(f"{self.colors['yellow']}Available modes: {', '.join(self.modes.keys())}{self.colors['reset']}")
        
        elif cmd == "/history":
            if not self.session_history:
                print(f"{self.colors['yellow']}No conversation history yet.{self.colors['reset']}")
            else:
                print(f"\n{self.colors['cyan']} CONVERSATION HISTORY:{self.colors['reset']}")
                print("=" * 50)
                for i, entry in enumerate(self.session_history[-5:], 1):  # Show last 5
                    print(f"{entry['timestamp']} [{entry['mode']}] Q: {entry['question'][:50]}...")
                    print(f"{'':>15} A: {entry['response'][:100]}...")
                    print(f"{'':>15}  {entry['response_time']:.2f}s\n")
        
        elif cmd == "/clear":
            self.session_history = []
            print(f"{self.colors['green']} Conversation history cleared.{self.colors['reset']}")
        
        elif cmd == "/health":
            # Check both servers
            print(f"{self.colors['cyan']}Checking server status...{self.colors['reset']}")
            print()
            
            # Check old medical API server
            old_server = self.check_server_health()
            if not old_server:
                print(f"{self.colors['yellow']}✗ Medical API Server (port 5050): Not running{self.colors['reset']}")
                print(f"   Start with: python Langapproach.py")
            print()
            
            # Check provisional diagnosis server
            provisional_server = self.check_provisional_server()
            if provisional_server:
                try:
                    response = requests.get(f"{PROVISIONAL_SERVER_URL}/health", timeout=5)
                    if response.status_code == 200:
                        health_data = response.json()
                        print(f"{self.colors['green']}✓ Provisional Diagnosis Server (port 5051): Running{self.colors['reset']}")
                        print(f"   Model: {health_data.get('model', 'N/A')}")
                        print(f"   Max iterations: {health_data.get('max_iterations', 'N/A')}")
                        print(f"   Num diagnoses: {health_data.get('num_diagnoses', 'N/A')}")
                    else:
                        print(f"{self.colors['yellow']}✗ Provisional Diagnosis Server (port 5051): Not responding{self.colors['reset']}")
                except:
                    print(f"{self.colors['yellow']}✗ Provisional Diagnosis Server (port 5051): Not running{self.colors['reset']}")
            else:
                print(f"{self.colors['yellow']}✗ Provisional Diagnosis Server (port 5051): Not running{self.colors['reset']}")
                print(f"   Start with: python provisional_diagnosis_system.py --server")
            print()
        
        elif cmd == "/file":
            self.selected_file = self.select_assessment_file()
            if self.selected_file:
                print(f"{self.colors['green']} Selected: {self.selected_file.name}{self.colors['reset']}")
                if self.current_mode == "provisional":
                    self.run_provisional_diagnosis(self.selected_file)
                else:
                    self.process_single_assessment(self.selected_file)
        
        elif cmd == "/provisional":
            # Check if server is running
            if not self.check_provisional_server():
                print(f"{self.colors['red']} Provisional diagnosis server is not running.{self.colors['reset']}")
                print(f"{self.colors['yellow']} Please start it first:{self.colors['reset']}")
                print(f"{self.colors['cyan']}   python provisional_diagnosis_system.py --server{self.colors['reset']}")
                print(f"{self.colors['yellow']} Server URL: {PROVISIONAL_SERVER_URL}{self.colors['reset']}")
                return True
            
            # Switch to provisional mode
            self.current_mode = "provisional"
            print(f"{self.colors['green']} Switched to: {self.modes['provisional']['description']}{self.colors['reset']}")
            print(f"{self.colors['green']} Server connected: {PROVISIONAL_SERVER_URL}{self.colors['reset']}")
            
            # Select and process file
            self.selected_file = self.select_assessment_file()
            if self.selected_file:
                print(f"{self.colors['green']} Selected: {self.selected_file.name}{self.colors['reset']}")
                self.run_provisional_diagnosis(self.selected_file)
            else:
                print(f"{self.colors['yellow']} No file selected.{self.colors['reset']}")
        
        elif cmd == "/batch":
            self.batch_process_assessments()
        
        elif cmd == "/help":
            self.print_header()
        
        else:
            print(f"{self.colors['red']}❌ Unknown command: {command}{self.colors['reset']}")
            print(f"   Type /help to see available commands.")
        
        return True
    
    def write_output_file(self, user_id: str, payload: dict):
        """Write structured output JSON to outputs directory."""
        try:
            # Create a subfolder per user_id and write the JSON inside it
            user_dir = self.outputs_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            out_path = user_dir / f"{user_id}.json"
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"{self.colors['green']} Saved: {out_path}{self.colors['reset']}")
            # If explanation report is present, save it as a sibling file
            if isinstance(payload, dict) and payload.get("explanation_report"):
                exp_path = user_dir / f"{user_id}_explanation.json"
                with open(exp_path, "w") as ef:
                    json.dump(payload["explanation_report"], ef, indent=2)
                print(f"{self.colors['green']} Saved explanation: {exp_path}{self.colors['reset']}")
        except Exception as e:
            print(f"{self.colors['red']} Failed to save output for {user_id}: {e}{self.colors['reset']}")
    
    def process_single_assessment(self, file_path: Path):
        """Process a single assessment file with structured endpoint and save output."""
        # Handle provisional mode separately
        if self.current_mode == "provisional":
            self.run_provisional_diagnosis(file_path)
            return
        
        # Check if old server is available (silent check)
        old_server_available = self.check_server_health(silent=True)
        
        # If old server not available, check if provisional server is available
        if not old_server_available:
            provisional_available = self.check_provisional_server()
            if provisional_available:
                print(f"{self.colors['yellow']} Medical API server (port 5050) is not running.{self.colors['reset']}")
                print(f"{self.colors['cyan']} Switching to provisional diagnosis mode...{self.colors['reset']}")
                self.current_mode = "provisional"
                self.run_provisional_diagnosis(file_path)
                return
            else:
                print(f"{self.colors['red']} No servers are running.{self.colors['reset']}")
                print(f"{self.colors['yellow']} Options:{self.colors['reset']}")
                print(f"   1. Start medical API server: python Langapproach.py")
                print(f"   2. Start provisional server: python provisional_diagnosis_system.py --server")
                return
        
        # Old server is available, proceed with normal processing
        try:
            with open(file_path) as f:
                data = json.load(f)
            user_id = file_path.stem
            
            # Use langgraph endpoint if langgraph mode is selected, otherwise use diagnose_assessment
            if self.current_mode == "langgraph":
                endpoint = self.base_url + "/langgraph_diagnose"
            else:
                endpoint = self.base_url + "/diagnose_assessment"
            
            payload = {"assessment_json": data, "user_id": user_id}
            print(f"{self.colors['yellow']} Processing {file_path.name}...{self.colors['reset']}")
            # Robust retry loop to tolerate slow LLM/backoffs on the server side
            attempts = 6
            last_err = None
            for i in range(attempts):
                try:
                    resp = requests.post(endpoint, json=payload, timeout=300)
                    if resp.status_code == 200:
                        out = resp.json()
                        self.write_output_file(user_id, out)
                        print(f"{self.colors['green']} Top diagnoses:{self.colors['reset']}")
                        # Handle both old format (llm_diagnosis) and new format (diagnoses)
                        if "diagnoses" in out:
                            for i, diag in enumerate(out["diagnoses"][:3], 1):
                                print(f"  {i}. {diag.get('diagnosis', 'N/A')} ({diag.get('probability', 0)}%)")
                        else:
                            print(out.get("llm_diagnosis", out.get("answer", ""))[:500])
                        break
                    else:
                        print(f"{self.colors['red']} Error {resp.status_code}: {resp.text}{self.colors['reset']}")
                        # Retry only on 5xx
                        if 500 <= resp.status_code < 600 and i < attempts - 1:
                            import time as _t, random as _r
                            backoff = min(30, (2 ** i) + _r.uniform(0, 1))
                            print(f"{self.colors['yellow']} Retrying in {backoff:.1f}s...{self.colors['reset']}")
                            _t.sleep(backoff)
                            continue
                        break
                except requests.exceptions.RequestException as e:
                    last_err = e
                    if i < attempts - 1:
                        import time as _t, random as _r
                        backoff = min(30, (2 ** i) + _r.uniform(0, 1))
                        print(f"{self.colors['yellow']} Connection issue: {e}. Retrying in {backoff:.1f}s...{self.colors['reset']}")
                        _t.sleep(backoff)
                        continue
                    else:
                        raise
        except Exception as e:
            print(f"{self.colors['red']} Error processing {file_path.name}: {e}{self.colors['reset']}")
    
    def batch_process_assessments(self):
        """Batch process all assessment JSON files and save EXAMPLE_OUTPUT-style results."""
        if not self.assessments_dir.exists():
            print(f"{self.colors['red']}Assessment directory not found: {self.assessments_dir}{self.colors['reset']}")
            return
        files = sorted([p for p in self.assessments_dir.glob("*.json")])
        if not files:
            print(f"{self.colors['yellow']}No assessment files found in {self.assessments_dir}{self.colors['reset']}")
            return
        print(f"{self.colors['yellow']} Processing {len(files)} files...{self.colors['reset']}")
        processed = 0
        failures = 0
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
                user_id = fp.stem  # derive from filename
                endpoint = self.base_url + "/diagnose_assessment"
                payload = {"assessment_json": data, "user_id": user_id}
                # Retry with backoff for batch as well
                attempts = 5
                for i in range(attempts):
                    try:
                        resp = requests.post(endpoint, json=payload, timeout=300)
                        if resp.status_code == 200:
                            out = resp.json()
                            self.write_output_file(user_id, out)
                            processed += 1
                            break
                        else:
                            print(f"{self.colors['red']} Error {resp.status_code} for {fp.name}{self.colors['reset']}")
                            if 500 <= resp.status_code < 600 and i < attempts - 1:
                                import time as _t, random as _r
                                backoff = min(30, (2 ** i) + _r.uniform(0, 1))
                                print(f"{self.colors['yellow']} Retrying {fp.name} in {backoff:.1f}s...{self.colors['reset']}")
                                _t.sleep(backoff)
                                continue
                            failures += 1
                            break
                    except requests.exceptions.RequestException as e:
                        if i < attempts - 1:
                            import time as _t, random as _r
                            backoff = min(30, (2 ** i) + _r.uniform(0, 1))
                            print(f"{self.colors['yellow']} Connection issue for {fp.name}: {e}. Retrying in {backoff:.1f}s...{self.colors['reset']}")
                            _t.sleep(backoff)
                            continue
                        print(f"{self.colors['red']} Failed {fp.name}: {e}{self.colors['reset']}")
                        failures += 1
                        break
            except Exception as e:
                print(f"{self.colors['red']} Failed {fp.name}: {e}{self.colors['reset']}")
                failures += 1
        print(f"{self.colors['cyan']} Batch complete. Success: {processed}, Failures: {failures}{self.colors['reset']}")
    
    def select_initial_mode(self):
        """Select between generic prompt or file-based assessment."""
        print(f"\n{self.colors['cyan']}{self.colors['bold']}SELECT MODE:{self.colors['reset']}")
        options = [
            "Generic Medical Prompt",
            "Patient Assessment File Analysis"
        ]
        menu = TerminalMenu(options, title="Use arrow keys to select:")
        choice = menu.show()
        return choice
    
    def select_assessment_file(self):
        """Select a patient assessment file from the directory."""
        if not self.assessments_dir.exists():
            print(f"{self.colors['red']}Assessment directory not found: {self.assessments_dir}{self.colors['reset']}")
            return None
        
        files = sorted([f.name for f in self.assessments_dir.glob("*.json")])
        if not files:
            print(f"{self.colors['red']}No assessment files found{self.colors['reset']}")
            return None
        
        print(f"\n{self.colors['cyan']}{self.colors['bold']}SELECT PATIENT FILE:{self.colors['reset']}")
        menu = TerminalMenu(files, title=f"Found {len(files)} files - Use arrow keys:")
        choice = menu.show()
        
        if choice is not None:
            return self.assessments_dir / files[choice]
        return None
    
    def load_assessment_file(self, file_path):
        """Load and format assessment file for diagnosis."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Load prompt template
            template_path = Path("/Users/chrisdev/Healthflex/mcp-clinicagent/dec-tree/src/AGENT_PROMPT_TEMPLATE.txt")
            if template_path.exists():
                with open(template_path) as f:
                    template = f.read()
                prompt = template.replace("{json_data}", json.dumps(data, indent=2))
            else:
                prompt = f"Analyze this patient assessment and provide differential diagnoses:\n\n{json.dumps(data, indent=2)}"
            
            return prompt
        except Exception as e:
            print(f"{self.colors['red']}Error loading file: {e}{self.colors['reset']}")
            return None
    
    def check_provisional_server(self):
        """Check if provisional diagnosis server is running."""
        try:
            response = requests.get(f"{PROVISIONAL_SERVER_URL}/health", timeout=5)
            if response.status_code == 200:
                return True
            return False
        except requests.exceptions.RequestException:
            return False
    
    def run_provisional_diagnosis(self, file_path: Path):
        """Run the provisional diagnosis system via HTTP server."""
        try:
            # Check if server is running
            if not self.check_provisional_server():
                print(f"{self.colors['red']} Provisional diagnosis server is not running.{self.colors['reset']}")
                print(f"{self.colors['yellow']} Please start it with: python provisional_diagnosis_system.py --server{self.colors['reset']}")
                print(f"{self.colors['yellow']} Server URL: {PROVISIONAL_SERVER_URL}{self.colors['reset']}")
                return
            
            print(f"\n{self.colors['cyan']}{self.colors['bold']}🔬 PROVISIONAL DIAGNOSIS SYSTEM{self.colors['reset']}")
            print("=" * 80)
            print(f"{self.colors['yellow']} Processing: {file_path.name}{self.colors['reset']}")
            print(f"{self.colors['yellow']} Using o1-mini reasoning model with MCP validation{self.colors['reset']}")
            print(f"{self.colors['yellow']} Connecting to server: {PROVISIONAL_SERVER_URL}{self.colors['reset']}")
            print(f"{self.colors['yellow']} This may take several minutes...{self.colors['reset']}")
            print("=" * 80)
            
            # Prepare request
            # Use absolute path for file
            abs_path = str(file_path.resolve())
            payload = {
                "json_path": abs_path
            }
            
            # Send request to server
            start_time = time.time()
            response = requests.post(
                f"{PROVISIONAL_SERVER_URL}/diagnose",
                json=payload,
                timeout=1800  # 30 minutes timeout
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "success":
                    report_data = result.get("report", {})
                    diagnoses = report_data.get("diagnoses", [])
                    
                    # Display results
                    print(f"\n{self.colors['green']}{self.colors['bold']}✓ DIAGNOSIS COMPLETE{self.colors['reset']}")
                    print("=" * 80)
                    print(f"{self.colors['cyan']}Patient ID: {report_data.get('patient_id', 'N/A')}{self.colors['reset']}")
                    print(f"{self.colors['cyan']}Generated {len(diagnoses)} provisional diagnoses:{self.colors['reset']}\n")
                    
                    for i, diag in enumerate(diagnoses, 1):
                        diag_name = diag.get("diagnosis_name", "Unknown")
                        confidence = diag.get("confidence_score", 0.0)
                        iterations = diag.get("iteration_count", 0)
                        reasoning = diag.get("reasoning", "")[:200]
                        fields = diag.get("json_fields_used", [])
                        
                        print(f"{self.colors['bold']}{i}. {diag_name}{self.colors['reset']}")
                        print(f"   {self.colors['yellow']}Confidence: {confidence:.2%}{self.colors['reset']}")
                        print(f"   {self.colors['cyan']}Iterations: {iterations}{self.colors['reset']}")
                        print(f"   {self.colors['purple']}Reasoning: {reasoning}...{self.colors['reset']}")
                        if fields:
                            print(f"   {self.colors['green']}Fields used: {', '.join(fields[:3])}{self.colors['reset']}")
                        print()
                    
                    print(f"{self.colors['yellow']}Processing time: {response_time:.2f}s{self.colors['reset']}")
                    print("=" * 80)
                    
                    # Save to history
                    self.session_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "question": f"Provisional diagnosis for {file_path.name}",
                        "mode": "provisional",
                        "response": f"Generated {len(diagnoses)} diagnoses",
                        "response_time": response_time
                    })
                    
                    # Output file is already saved by the server
                    user_id = file_path.stem
                    output_dir = self.outputs_dir / user_id
                    print(f"{self.colors['green']}✓ Report saved to: {output_dir}/{self.colors['reset']}")
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"{self.colors['red']} Server returned error: {error_msg}{self.colors['reset']}")
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                error_msg = error_data.get("error", f"HTTP {response.status_code}")
                print(f"{self.colors['red']} Error from server: {error_msg}{self.colors['reset']}")
                
        except requests.exceptions.Timeout:
            print(f"{self.colors['red']} Request timed out. The diagnosis may take longer than expected.{self.colors['reset']}")
        except requests.exceptions.RequestException as e:
            print(f"{self.colors['red']} Error connecting to server: {e}{self.colors['reset']}")
            print(f"{self.colors['yellow']} Make sure the server is running: python provisional_diagnosis_system.py --server{self.colors['reset']}")
        except Exception as e:
            print(f"{self.colors['red']} Error running provisional diagnosis: {e}{self.colors['reset']}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the interactive chat interface."""
        self.print_header()
        
        # Check server health at startup (optional - only warn, don't block)
        old_server_available = self.check_server_health(silent=True)
        provisional_server_available = self.check_provisional_server()
        
        if not old_server_available and not provisional_server_available:
            print(f"{self.colors['yellow']}Note: No servers are currently running.{self.colors['reset']}")
            print(f"{self.colors['yellow']}  - Medical API server (port 5050): Start with 'python Langapproach.py'{self.colors['reset']}")
            print(f"{self.colors['yellow']}  - Provisional server (port 5051): Start with 'python provisional_diagnosis_system.py --server'{self.colors['reset']}")
            print()
        elif not old_server_available:
            print(f"{self.colors['yellow']}Note: Medical API server (port 5050) is not running.{self.colors['reset']}")
            print(f"{self.colors['green']}✓ Provisional diagnosis server (port 5051) is available.{self.colors['reset']}")
            print(f"{self.colors['cyan']}You can use /provisional mode or it will auto-switch when processing files.{self.colors['reset']}")
            print()
        elif not provisional_server_available:
            print(f"{self.colors['green']}✓ Medical API server (port 5050) is running.{self.colors['reset']}")
            print(f"{self.colors['yellow']}Note: Provisional diagnosis server (port 5051) is not running.{self.colors['reset']}")
            print()
        else:
            print(f"{self.colors['green']}✓ Both servers are running and available.{self.colors['reset']}")
            print()
        
        # Select initial mode
        mode_choice = self.select_initial_mode()
        
        if mode_choice == 1:  # Patient Assessment File
            self.selected_file = self.select_assessment_file()
            if self.selected_file:
                print(f"\n{self.colors['green']} Selected: {self.selected_file.name}{self.colors['reset']}")
                print(f"{self.colors['yellow']} Analyzing patient assessment...{self.colors['reset']}")
                self.process_single_assessment(self.selected_file)
            else:
                print(f"{self.colors['yellow']}No file selected. Switching to generic mode.{self.colors['reset']}")
        
        print(f"\n{self.colors['green']} Ready! Start asking medical questions...{self.colors['reset']}")
        print(f"{self.colors['yellow']} Tip: Type /file to analyze another assessment file{self.colors['reset']}")
        
        while True:
            try:
                # Get current mode color
                mode_color = self.modes[self.current_mode]["color"]
                
                # Input prompt
                user_input = input(f"\n{mode_color}[{self.current_mode}] 🏥 You: {self.colors['reset']}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Send medical question
                self.send_question(user_input)
                
            except KeyboardInterrupt:
                print(f"\n\n{self.colors['yellow']} Chat interrupted. Goodbye!{self.colors['reset']}")
                break
            except EOFError:
                print(f"\n\n{self.colors['yellow']} Chat ended. Goodbye!{self.colors['reset']}")
                break
        
        print(f"{self.colors['cyan']}Total questions asked this session: {len(self.session_history)}{self.colors['reset']}")

if __name__ == "__main__":
    # Check if server URL is provided as argument
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5050"
    
    # Create and run chat interface
    chat = MedicalChatInterface(server_url)
    chat.run()