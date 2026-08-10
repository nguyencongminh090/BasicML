#!/usr/bin/env bash
# Allocate the next ID for the ai-audit workspace.
# Usage: next_id.sh {todo|fix|convo}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-}" in
  todo)  DIR="$ROOT/todo";    PREFIX="TODO"  ;;
  fix)   DIR="$ROOT/fix-log"; PREFIX="FIX"   ;;
  convo) DIR="$ROOT/convo";   PREFIX="CONVO" ;;
  *)
    echo "Usage: $0 {todo|fix|convo}" >&2
    exit 1
    ;;
esac

last=$(grep -rhoE "${PREFIX}-[0-9]{4}" "$DIR" 2>/dev/null | sed -E "s/${PREFIX}-//" | sort -n | tail -1) || true
last=${last:-0}
next=$((10#$last + 1))

printf "%s-%04d\n" "$PREFIX" "$next"
