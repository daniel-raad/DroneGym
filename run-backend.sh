#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/backend"
pip3 install -q -r requirements.txt
exec python3 -W ignore -m uvicorn app.main:app --reload --port 8000
