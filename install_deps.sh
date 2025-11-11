#!/bin/bash

echo "Installing dependencies for MCP Clinic Agent..."

python3 -m venv venv-agent
source venv-agent/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Dependencies installed successfully!"
