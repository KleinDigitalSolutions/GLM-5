#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  echo "Warnung: $ENV_FILE existiert bereits."
  read -r -p "Ueberschreiben? (y/N): " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Abgebrochen."
    exit 0
  fi
fi

read -r -s -p "Modal API Key eingeben: " MODAL_API_KEY
echo

if [[ -z "${MODAL_API_KEY}" ]]; then
  echo "Fehler: API Key ist leer."
  exit 1
fi

cat > "$ENV_FILE" <<EOT
MODAL_API_KEY=${MODAL_API_KEY}
MODEL_ID=zai-org/GLM-5-FP8
MODAL_BASE_URL=https://api.us-west-2.modal.direct/v1
EOT

chmod 600 "$ENV_FILE"

echo "Fertig: $ENV_FILE wurde erstellt (Berechtigung 600)."
echo "Naechster Schritt: ./chat.sh \"Hallo\""
