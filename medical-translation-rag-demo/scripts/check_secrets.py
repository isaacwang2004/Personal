"""Small local guardrail; use a real secret scanner in CI for production repos."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "likely OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Databricks token": re.compile(r"\bdapi[A-Za-z0-9_-]{20,}\b"),
    "hard-coded password": re.compile(r"(?i)password\s*=\s*['\"][^'\"]{8,}['\"]"),
    "hard-coded secret": re.compile(r"(?i)(?:api_key|secret|token)\s*=\s*['\"][^'\"]{12,}['\"]"),
}

ignore = {".git", ".venv", "__pycache__"}
hits = []
for path in ROOT.rglob("*"):
    if not path.is_file() or any(part in ignore for part in path.parts):
        continue
    if path.name == ".env.example":
        continue
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        continue
    for name, pattern in PATTERNS.items():
        if pattern.search(text):
            hits.append((path.relative_to(ROOT), name))

if hits:
    for path, name in hits:
        print(f"Potential secret: {name}: {path}")
    sys.exit(1)

print("No obvious secrets detected.")
