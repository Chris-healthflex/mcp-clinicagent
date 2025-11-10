#!/usr/bin/env python3
"""
Interactive Medical Chat Terminal
Chat interface to interact with the LangChain Medical API Server with MCP Tools
"""

import requests
import json
import sys
from datetime import datetime
import time

class MedicalChatInterface:
    """Interactive chat interface for the medical API."""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session_history = []
        self.current_mode = "intelligent"  # intelligent, retrieval, direct
        
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
        print(f"{self.colors['bold']}{self.colors['blue']}")
        print("=" * 80)
        print(" INTERACTIVE MEDICAL CHAT TERMINAL")
        print(" Connected to LangChain Medical API with MCP Tools")
        print("=" * 80)
        print(f"{self.colors['reset']}")
        
        print(f"{self.colors['cyan']}Available Modes:{self.colors['reset']}")
        for mode, info in self.modes.items():
            marker = "." if mode == self.current_mode else "  "
            print(f"{marker} {mode}: {info['description']}")
        
        print(f"\n{self.colors['yellow']}Commands:{self.colors['reset']}")
        print("  /mode <mode_name>  - Switch between modes")
        print("  /history          - View conversation history") 
        print("  /clear            - Clear conversation history")
        print("  /health           - Check API server status")
        print("  /help             - Show this help")
        print("  /quit or /exit    - Exit the chat")
        print("=" * 80)
    
    def check_server_health(self):
        """Check if the API server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=30)
            if response.status_code == 200:
                health_data = response.json()
                print(f"{self.colors['green']} Server Status: {health_data.get('status', 'unknown')}{self.colors['reset']}")
                print(f"   MCP Status: {health_data.get('mcp_status', 'unknown')}")
                print(f"   Database: {health_data.get('database_path', 'unknown')}")
                return True
            else:
                print(f"{self.colors['red']} Server returned status: {response.status_code}{self.colors['reset']}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"{self.colors['red']} Cannot connect to server: {e}{self.colors['reset']}")
            print(f"   Make sure the server is running at {self.base_url}")
            return False
    
    def send_question(self, question):
        """Send question to the appropriate API endpoint."""
        mode_info = self.modes[self.current_mode]
        endpoint = self.base_url + mode_info["endpoint"]
        
        try:
            print(f"{self.colors['yellow']} Processing with {mode_info['description']}...{self.colors['reset']}")
            
            # Prepare request based on mode
            if self.current_mode == "direct":
                # For direct mode, use the same format as other modes
                payload = {"question": question}
            else:
                payload = {"question": question}
            
            # Send request
            start_time = time.time()
            response = requests.post(endpoint, json=payload, timeout=120)
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
        
        # Main answer
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
            self.check_server_health()
        
        elif cmd == "/help":
            self.print_header()
        
        else:
            print(f"{self.colors['red']}❌ Unknown command: {command}{self.colors['reset']}")
            print(f"   Type /help to see available commands.")
        
        return True
    
    def run(self):
        """Run the interactive chat interface."""
        self.print_header()
        
        # Check server health at startup
        if not self.check_server_health():
            print(f"{self.colors['red']}Please start the medical API server first:{self.colors['reset']}")
            print("python Langapproach.py")
            return
        
        print(f"\n{self.colors['green']} Ready! Start asking medical questions...{self.colors['reset']}")
        print(f"{self.colors['yellow']} Tip: Try 'What are the symptoms of diabetes?' or 'How is hypertension treated?'{self.colors['reset']}")
        
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
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    # Create and run chat interface
    chat = MedicalChatInterface(server_url)
    chat.run()