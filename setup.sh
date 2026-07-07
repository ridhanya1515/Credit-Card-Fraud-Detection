#!/usr/bin/env bash
# Setup script for Credit Card Fraud Detection project.
# Usage:
#   bash setup.sh

set -e

echo "Installing dependencies and preparing project..."

if [ -d "venv" ]; then
  echo "Virtual environment already exists. Activate it with: source venv/bin/activate"
else
  python3 -m venv venv
  echo "Created virtual environment in ./venv"
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing project dependencies..."
pip install -r requirements.txt

echo "Installation complete. To activate the environment, run: source venv/bin/activate"

echo "If you still need the Kaggle dataset, download creditcard.csv from:"
echo "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
echo "Place it into the data/ directory before running train.py or app.py"
