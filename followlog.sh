#!/usr/bin/env bash
# Print a log file then follow it as it's appended to.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path-to-log-file>" >&2
    exit 1
fi

tail -f -n +1 "$1"
