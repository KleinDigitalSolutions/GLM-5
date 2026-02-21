#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Fehler: $ENV_FILE fehlt. Starte zuerst ./setup.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${MODAL_API_KEY:-}" ]]; then
  echo "Fehler: MODAL_API_KEY ist nicht gesetzt in $ENV_FILE"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Fehler: jq ist nicht installiert. Installiere es mit: brew install jq"
  exit 1
fi

PROMPT="${*:-How many r-s are in strawberry?}"

PAYLOAD="$(jq -n \
  --arg model "${MODEL_ID:-zai-org/GLM-5-FP8}" \
  --arg prompt "$PROMPT" \
  '{model: $model, messages: [{role: "user", content: $prompt}], max_tokens: 500}')"

RESPONSE="$(curl -sS -X POST "${MODAL_BASE_URL:-https://api.us-west-2.modal.direct/v1}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${MODAL_API_KEY}" \
  -d "$PAYLOAD")"

echo "$RESPONSE" | jq -r '
  .choices[0].message as $m
  | if (($m.reasoning_content // "") | length) > 0 then
      "Denken:\n\($m.reasoning_content)\n\nAntwort:\n\($m.content // "")"
    else
      "Antwort:\n\($m.content // "")"
    end
'
