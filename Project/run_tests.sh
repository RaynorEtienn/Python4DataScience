#!/bin/bash
# Script to run tests using the project's virtual environment

# Get the absolute path to the project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="$PROJECT_ROOT/../.venv/bin/python"

if [ -f "$VENV_PYTHON" ]; then
    echo "Running tests using $VENV_PYTHON..."
    "$VENV_PYTHON" -m unittest discover tests
else
    echo "Error: Virtual environment python not found at $VENV_PYTHON"
    echo "Please ensure the .venv exists in the parent directory."
    exit 1
fi
