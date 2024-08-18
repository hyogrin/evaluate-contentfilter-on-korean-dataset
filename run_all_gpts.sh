#!/bin/bash

# Copy specific environment file to .env
cp .env_gpt-4o .env

# Run the content filter evaluations
python3 main_multiprocessing.py

# gpt-4o-mini
cp .env_gpt-4o-mini .env

python3 main_multiprocessing.py
