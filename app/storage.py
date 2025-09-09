import uuid
from pathlib import Path
from .config import settings

def save_upload_file_bytes(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix or ""
    fname = f"{uuid.uuid4().hex}{ext}"
    path = Path(settings.TEMP_DIR) / fname
    path.write_bytes(content)
    return str(path)

def cleanup_file(path: str):
    try:
        Path(path).unlink()
    except Exception:
        pass
