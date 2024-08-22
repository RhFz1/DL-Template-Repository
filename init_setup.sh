#!/bin/bash
echo "$(date): Setting up the environment"

echo "$(date): Creating the virtual environment, with python3.10"

python3.10 -m venv venv

echo "$(date): Activating the virtual environment"

source venv/bin/activate

echo "$(date): Upgrading pip"

pip install --upgrade pip

echo "$(date): Installing the dev requirements"

pip install -r requirements-dev.txt