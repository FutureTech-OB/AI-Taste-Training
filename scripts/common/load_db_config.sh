#!/bin/bash

load_db_config() {
  local repo_root="$1"
  local requested_db_name="${2:-}"
  local config_path="${repo_root}/assets/database.toml"

  if [ ! -f "${config_path}" ]; then
    echo "database.toml not found: ${config_path}" >&2
    return 1
  fi

  local output
  output=$(REPO_ROOT="${repo_root}" REQUESTED_DB_NAME="${requested_db_name}" python - <<'PY'
import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

repo_root = Path(os.environ["REPO_ROOT"])
config_path = repo_root / "assets" / "database.toml"
requested_db_name = os.environ.get("REQUESTED_DB_NAME", "").strip()

with open(config_path, "rb") as f:
    config = tomllib.load(f)

default_cfg = config.get("default", {})
db_name = requested_db_name or default_cfg.get("db_name", "RQ")
conn_str = default_cfg.get("connection_string", "")
if not conn_str:
    raise SystemExit("database.toml missing default.connection_string")

connection_string = conn_str.replace("<DBNAME>", db_name) if "<DBNAME>" in conn_str else conn_str
print(db_name)
print(connection_string)
PY
)

  if [ $? -ne 0 ] || [ -z "${output}" ]; then
    return 1
  fi

  DB_NAME=$(printf '%s\n' "${output}" | sed -n '1p')
  MONGODB_CONNECTION_STRING=$(printf '%s\n' "${output}" | sed -n '2p')
  export DB_NAME
  export MONGODB_CONNECTION_STRING
}
