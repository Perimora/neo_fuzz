#!/bin/bash

set -e

# create venv
python3 -m venv .venv

# activate venv
source .venv/bin/activate

# update pip
pip install --upgrade pip

# pip requirements
pip install -r config/requirements/requirements-dev.txt

# install jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name neo_fuzz --display-name "Python (neo_fuzz)"
