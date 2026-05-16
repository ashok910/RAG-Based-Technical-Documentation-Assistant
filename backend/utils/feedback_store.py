"""
Simple JSON-file-based feedback store.
In production, replace with a database.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)
_lock = Lock()
FEEDBACK_FILE = Path("data/feedback.jsonl")


def save_feedback(data: dict) -> bool:
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {**data, "timestamp": datetime.utcnow().isoformat()}
        with _lock:
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False


def get_feedback_stats() -> dict:
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "up": 0, "down": 0}
    up = down = 0
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("rating") == "up":
                    up += 1
                else:
                    down += 1
            except Exception:
                pass
    return {"total": up + down, "up": up, "down": down}
