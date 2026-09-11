#!/bin/sh
# Fails (non-zero exit) if the API is not responding or the model isn't loaded.
set -e
STATUS=$(python3 -c "
import json, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) as r:
        body = json.loads(r.read())
        print('ok' if body.get('status') == 'ok' else 'fail')
except Exception:
    print('fail')
")
if [ "$STATUS" != "ok" ]; then
    exit 1
fi
exit 0
