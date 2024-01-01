#!/bin/bash

# we need python greater than 3.8
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")

echo "Python version: $python_version"

if [[ $python_version < "3.8" ]]; then
    echo "Error: Python 3.8 or higher is required."
    exit 1
fi

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created in '$VENV_DIR'."
fi

source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

python_location=$(which python3)
echo "Python location: $python_location"

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r build-requirements.txt
python3 -m pip install -r requirements.txt
chmod +x launcher.py