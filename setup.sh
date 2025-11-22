#!/bin/bash
echo "Creating environment..."
conda env create -f environment.yml
conda activate research-env

echo "Setup complete."
