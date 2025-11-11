#!/bin/bash

# Installation script for MCP Medical API Server
# Run this in your venv: bash install_dependencies.sh

echo "🔧 Installing dependencies for MCP Medical API Server..."
echo "=================================================="

# Activate venv (if not already activated)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected!"
    echo "Please activate your venv first:"
    echo "  source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

echo ""
echo "📥 Installing core dependencies..."

# Core Flask
pip install flask python-dotenv

# OpenAI
pip install openai

# Requests for HTTP calls
pip install requests

# CRITICAL: Install specific LangChain versions that work
echo ""
echo "🔗 Installing LangChain packages (compatible versions)..."

# Uninstall any existing langchain packages first
pip uninstall -y langchain langchain-openai langchain-community langchain-core langchain-text-splitters 2>/dev/null

# Install compatible versions
pip install "langchain==0.2.16"
pip install "langchain-openai==0.1.25"
pip install "langchain-community==0.2.16"
pip install "langchain-core==0.2.38"
pip install "langchain-text-splitters==0.2.4"

# Additional dependencies
pip install tiktoken  # For token counting
pip install pydantic==2.9.2  # Compatible version

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing imports..."

python3 << EOF
try:
    import flask
    print("✓ Flask imported successfully")
    
    import openai
    print("✓ OpenAI imported successfully")
    
    import requests
    print("✓ Requests imported successfully")
    
    # Test new import structure
    from langchain_core.prompts import PromptTemplate
    print("✓ LangChain prompts imported successfully")
    
    from langchain_openai import ChatOpenAI
    print("✓ LangChain OpenAI imported successfully")
    
    from langchain.chains import RetrievalQA
    print("✓ LangChain chains imported successfully")
    
    from langchain.agents import AgentType, initialize_agent
    print("✓ LangChain agents imported successfully")
    
    from langchain.memory import ConversationBufferMemory
    print("✓ LangChain memory imported successfully")
    
    from langchain_core.documents import Document
    print("✓ LangChain documents imported successfully")
    
    from langchain_core.retrievers import BaseRetriever
    print("✓ LangChain retrievers imported successfully")
    
    from langchain.tools import Tool
    print("✓ LangChain tools imported successfully")
    
    print("")
    print("🎉 All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

echo ""
echo "=================================================="
echo "✅ Setup complete! You can now run:"
echo "   python3 Langapproach_mcp_client.py"
echo "=================================================="