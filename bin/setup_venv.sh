#!/bin/bash
deactivate 2> /dev/null
rm -rf venv
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

