#!/usr/bin/env bash
# macOS / Linux: ./run_app.sh
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
